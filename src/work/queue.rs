use crate::work::task::{Task, TaskID};
use flume::Sender;
use tokio::task::block_in_place;

pub enum QueueCommand<T: Task> {
    Append(Box<T>),
    Delete(TaskID),
    Process,
    Clear,
    Stop,
}

#[derive(Debug)]
pub struct Queue<T: Task> {
    queue_sender: Sender<QueueCommand<T>>,
}

impl<T: Task> Queue<T> {
    pub fn new() -> Self {
        let (queue_sender, queue_receiver) = flume::unbounded::<QueueCommand<T>>();

        // Launch background thread
        tokio::spawn(async move {
            let mut tasks = Vec::new();

            while let Ok(command) = queue_receiver.recv_async().await {
                match command {
                    QueueCommand::Append(task) => {
                        let task = task.clone();
                        tracing::info!("Appending task {:?}", task);
                        tasks.push(task.clone());

                        // use the ThreadPool here instead
                        tokio::spawn(async move {
                            block_in_place(move || {
                                task.process(); // blocking process
                            });
                        });
                    }
                    QueueCommand::Delete(id) => {
                        tracing::info!("Deleting task {}", id);
                        tasks.retain(|task| task.get_id() != id);
                    }
                    QueueCommand::Clear => {
                        tracing::info!("Clearing all tasks");
                        tasks.clear();
                    }
                    QueueCommand::Process => {
                        tracing::info!("Processing tasks");
                    }
                    QueueCommand::Stop => {
                        tracing::info!("Stopping queue");
                        // Stop receiving new tasks
                        // Wait for all running tasks to finish
                        while (queue_receiver.recv_async().await).is_ok() {}
                        break;
                    }
                }
            }
        });
        Self { queue_sender }
    }
    pub(crate) fn append(&self, task: Box<T>) -> Result<(), flume::SendError<QueueCommand<T>>> {
        self.queue_sender.send(QueueCommand::Append(task))?;
        Ok(())
    }

    fn delete(&self, id: TaskID) -> Result<(), flume::SendError<QueueCommand<T>>> {
        self.queue_sender.send(QueueCommand::Delete(id))?;
        Ok(())
    }

    pub(crate) fn clear(&self) -> Result<(), flume::SendError<QueueCommand<T>>> {
        self.queue_sender.send(QueueCommand::Clear)?;
        Ok(())
    }

    pub(crate) fn process(&self) -> Result<(), flume::SendError<QueueCommand<T>>> {
        self.queue_sender.send(QueueCommand::Process)?;
        Ok(())
    }

    pub(crate) fn stop(&self) -> Result<(), flume::SendError<QueueCommand<T>>> {
        self.queue_sender.send(QueueCommand::Stop)?;
        Ok(())
    }

    pub(crate) async fn join(self) -> Result<(), tokio::task::JoinError> {
        self.stop().expect("Failed to stop queue");
        // self.worker_pool.join().await?;
        // self.join_handle.await?; // TODO: Why does this block?
        Ok(())
    }
}
