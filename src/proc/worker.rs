use std::thread::sleep;
use std::time::Duration;
use flume::Receiver;
use tokio::task::spawn_blocking;
use crate::queue::Task;

#[derive(Clone, Debug)]
pub struct Worker {
    pub receiver: Receiver<Box<Task>>,
}

impl Worker {
    pub(crate) fn new(receiver: Receiver<Box<Task>>) -> Self {
        let recv_clone = receiver.clone();
        spawn_blocking(move || {
            while let Ok(task) = recv_clone.recv() {
                tracing::info!("Processing task {:?}", task);
                sleep(Duration::from_secs(5));
                println!("Task {:?} processed", task);
            }
        });
        Self { receiver }
    }
}

struct WorkerPool {
    workers: Vec<Worker>,
}

impl WorkerPool {
    pub(crate) fn new(num_workers: usize, receiver: Receiver<Box<Task>>) -> Self {
        // Create set of proxy receivers
        let mut workers = Vec::new();
        for _ in 0..num_workers {
            let proxy_receiver = receiver.clone();
            let worker = Worker::new(proxy_receiver);
            workers.push(worker);
        }
        Self { workers }
    }
}

#[cfg(test)]
mod tests {
    use crate::queue::{Queue, SubmitTask};
    use crate::worker::WorkerPool;
    use crate::queue::Task;
    use std::thread::sleep;
    use std::time::Duration;

    #[tokio::test]
    async fn test_worker_pool() {
        let queue = Queue::new();
        // Connect receiver to queue's sender
        let receiver = queue.receiver.clone();

        let tasks = (0..10).map(|i| {
            SubmitTask { description: format!("Task {}", i) }.into()
        }).collect::<Vec<Task>>();

        sleep(Duration::from_secs(10));
    }
}