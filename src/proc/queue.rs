use std::sync::Arc;
use axum::extract::State;
use axum::Json;
use axum::http::StatusCode;
use flume::Sender;
use crate::proc::task::{Task, TaskID};
use crate::proc::worker::WorkerPool;

pub enum QueueCommand<T: Task> {
    Append(Box<T>),
    Delete(TaskID),
    Process,
    Clear,
    Stop,
}

#[derive(Debug)]
pub struct Queue<T: Task> {
    pub queue_sender: Sender<QueueCommand<T>>,
    pub worker_pool: WorkerPool<T>,
}

impl<T: Task + 'static> Queue<T> {
    pub(crate) fn new(n_workers: usize) -> Self {
        let (queue_sender, queue_receiver) = flume::unbounded::<QueueCommand<T>>();

        let (worker_sender, worker_receiver) = flume::bounded(n_workers);

        let worker_pool = WorkerPool::new(n_workers, worker_receiver);

        // Launch background thread
        tokio::spawn(async move {
            let mut tasks = Vec::new();
            while let Ok(command) = queue_receiver.recv_async().await {
                match command {
                    QueueCommand::Append(task) => {
                        tracing::info!("Appending task {:?}", task);
                        tasks.push(task.clone()); // TODO: remove clone
                        if (worker_sender.send_async(task).await).is_ok() {
                            tracing::info!("Task sent to worker");
                        } else {
                            tracing::error!("Failed to send task to worker");
                        }
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
                        drop(worker_sender);
                        // Wait for all running tasks to finish
                        while (queue_receiver.recv_async().await).is_ok() {}
                        break;
                    }
                }
            }
        });
        Self { queue_sender, worker_pool }
    }

    pub(crate) fn append(&self, task: Box<T>) -> Result<(), flume::SendError<QueueCommand<T>>> {
        self.queue_sender.send(QueueCommand::Append(task))?;
        Ok(())
    }

    fn delete(&self, id: TaskID) -> Result<(), flume::SendError<QueueCommand<T>>> {
        self.queue_sender.send(QueueCommand::Delete(id))?;
        Ok(())
    }

    fn clear(&self) -> Result<(), flume::SendError<QueueCommand<T>>> {
        self.queue_sender.send(QueueCommand::Clear)?;
        Ok(())
    }

    fn process(&self) -> Result<(), flume::SendError<QueueCommand<T>>> {
        self.queue_sender.send(QueueCommand::Process)?;
        Ok(())
    }

    pub(crate) fn stop(&self) -> Result<(), flume::SendError<QueueCommand<T>>> {
        self.queue_sender.send(QueueCommand::Stop)?;
        Ok(())
    }

    pub(crate) async fn join(self) -> Result<(), tokio::task::JoinError> {
        self.stop().expect("Failed to stop queue");
        self.worker_pool.join().await?;
        // self.join_handle.await?; // TODO: Why does this block?
        Ok(())
    }
}

pub async fn create_task<T: Task + 'static>(
    State(queue): State<Arc<Queue<T>>>,
    Json(submit): Json<T::Input>,
) -> StatusCode {
    tracing::info!("Creating a new task");
    if queue.append(Box::new(T::from_input(submit))).is_ok() {
        StatusCode::CREATED
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

// #[debug_handler]
// pub async fn delete_task(
//     State(state): State<Arc<Queue>>,
//     path: Path<TaskID>,
// ) -> StatusCode {
//     tracing::info!("Deleting task {}", path.0);
//     if let Err(_) = state.delete(path.0) {
//         StatusCode::NOT_FOUND
//     } else {
//         StatusCode::OK
//     }
// }

pub async fn clear_tasks<T: Task + 'static>(State(state): State<Arc<Queue<T>>>) -> StatusCode {
    tracing::info!("Clearing all tasks");
    if state.clear().is_err() {
        StatusCode::INTERNAL_SERVER_ERROR
    } else {
        StatusCode::OK
    }
}

pub async fn process_tasks<T: Task + 'static>(State(state): State<Arc<Queue<T>>>) -> StatusCode {
    tracing::info!("Processing tasks");
    if state.process().is_err() {
        StatusCode::INTERNAL_SERVER_ERROR
    } else {
        StatusCode::OK
    }
}