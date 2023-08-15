use flume::Receiver;
use tokio::task::{JoinHandle, spawn_blocking};
use crate::proc::task::{Task};

#[derive(Debug)]
pub struct Worker<T: Task> {
    pub receiver: Receiver<Box<T>>,
    join_handle: JoinHandle<()>,
}

impl<T: Task + 'static> Worker<T> {
    pub(crate) fn new(receiver: Receiver<Box<T>>) -> Self {
        // let (tx_state, rx_state) = flume::unbounded();
        let recv_clone = receiver.clone();
        let join_handle= spawn_blocking(move || {
            while let Ok(task) = recv_clone.recv() {
                task.process();
            }
        });
        Self { receiver, join_handle }
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
        let workers = (0..num_workers).map(|_| {
            let worker_receiver = receiver.clone();
            Worker::new(worker_receiver)
        }).collect::<Vec<_>>();
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
    use std::thread::sleep;
    use std::time::Duration;
    use crate::proc::queue::Queue;
    use crate::proc::task::{Task, TaskID};
    