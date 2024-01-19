use crate::work::task::Task;
use flume::Receiver;
use tokio::task::{spawn_blocking, JoinHandle};

#[derive(Debug)]
pub struct Worker<T: Task> {
    pub receiver: Receiver<Box<T>>,
    join_handle: JoinHandle<()>,
}

impl<T: Task + 'static> Worker<T> {
    pub(crate) fn new(receiver: Receiver<Box<T>>) -> Self {
        // let (tx_state, rx_state) = flume::unbounded();
        let recv_clone = receiver.clone();
        let join_handle = spawn_blocking(move || {
            while let Ok(task) = recv_clone.recv() {
                task.process();
            }
        });
        Self {
            receiver,
            join_handle,
        }
    }

    pub(crate) async fn join(self) -> Result<(), tokio::task::JoinError> {
        drop(self.receiver);
        self.join_handle.await?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct WorkerPool<T: Task> {
    workers: Vec<Worker<T>>,
}

impl<T: Task + 'static> WorkerPool<T> {
    pub(crate) fn new(num_workers: usize, receiver: Receiver<Box<T>>) -> Self {
        // Create set of proxy receivers
        let workers = (0..num_workers)
            .map(|_| {
                let worker_receiver = receiver.clone();
                Worker::new(worker_receiver)
            })
            .collect::<Vec<_>>();
        Self { workers }
    }

    pub(crate) async fn join(self) -> Result<(), tokio::task::JoinError> {
        for worker in self.workers {
            worker.join().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::work::tests::ExampleTask;
    use std::thread::sleep;
    use std::time::Duration;

    #[tokio::test]
    async fn test_worker() {
        // initialize tracing
        let subscriber = tracing_subscriber::FmtSubscriber::builder()
            .with_max_level(tracing::Level::TRACE) // Set the maximum tracing level
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .expect("setting default subscriber failed");

        let (tx, rx) = flume::unbounded();
        let worker = super::Worker::new(rx);
        let task = Box::new(ExampleTask { input: "Test".into() });
        tx.send_async(task).await.expect("Failed to send task");
        sleep(Duration::from_millis(500));
        drop(tx);
        worker.join().await.expect("Failed to join worker");
    }
}
