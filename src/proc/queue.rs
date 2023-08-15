use axum::extract::State;
use axum::Json;
use axum::http::StatusCode;
use futures_util::future::join_all;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use crate::worker::Worker;

#[derive(Deserialize, Serialize)]
pub struct SubmitTask {
    pub(crate) description: String,
}

impl From<SubmitTask> for Task {
    fn from(submit: SubmitTask) -> Self {
        Task {
            id: Uuid::new_v4(),
            description: submit.description,
            done: false,
        }
    }
}

pub type TaskID = Uuid;

#[derive(Clone, Debug)]
pub struct Task {
    pub id: TaskID,
    pub description: String,
    pub done: bool,
}

/// Dummy processing function that will just sleep for 5 seconds.
async fn process_task(task: Box<Task>) {
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    tracing::info!("Task {:?} processed", task);
}

pub enum QueueCommand {
    Append(Box<Task>),
    Delete(TaskID),
    Process,
    Clear,
}

#[derive(Clone, Debug)]
pub struct Queue {
    pub queue_sender: flume::Sender<QueueCommand>,
    pub worker_pool: Vec<Worker>,
}

impl Queue {
    pub(crate) fn new(n_workers: usize) -> Self {
        let (queue_sender, queue_receiver) = flume::unbounded();

        let (worker_sender, worker_receiver) = flume::bounded(n_workers);

        let worker_pool = (0..n_workers).map(|_| {
            let worker_receiver = worker_receiver.clone();
            Worker::new(worker_receiver)
        }).collect::<Vec<_>>();


        // Launch background thread
        tokio::spawn(async move {
            let mut tasks = Vec::new();
            while let Ok(command) = queue_receiver.recv_async().await {
                match command {
                    QueueCommand::Append(task) => {
                        tracing::info!("Appending task {:?}", task);
                        tasks.push(task.clone()); // TODO: remove clone
                        if let Ok(_) = worker_sender.send_async(task).await {
                            tracing::info!("Task sent to worker");
                        } else {
                            tracing::error!("Failed to send task to worker");
                        }
                    }
                    QueueCommand::Delete(id) => {
                        tracing::info!("Deleting task {}", id);
                        tasks.retain(|task| task.id != id);
                    }
                    QueueCommand::Clear => {
                        tracing::info!("Clearing all tasks");
                        tasks.clear();
                    }
                    QueueCommand::Process => {
                        tracing::info!("Processing tasks");
                    }
                }
            }
        });
        Self { queue_sender, worker_pool }
    }

    pub(crate) fn append(&self, task: Box<Task>) {
        tracing::info!("Appending task {:?}", task);
    }

    fn delete(&self, id: TaskID) -> Result<(), flume::SendError<QueueCommand>> {
        tracing::info!("Deleting task {}", id);
        self.queue_sender.send(QueueCommand::Delete(id))?;
        Ok(())
    }

    fn clear(&self) -> Result<(), flume::SendError<QueueCommand>> {
        tracing::info!("Clearing all tasks");
        self.queue_sender.send(QueueCommand::Clear)?;
        Ok(())
    }

    fn process(&self) -> Result<(), flume::SendError<QueueCommand>> {
        tracing::info!("Processing tasks");
        self.queue_sender.send(QueueCommand::Process)?;
        Ok(())
    }
}

pub async fn create_task(
    State(state): State<Queue>,
    Json(submit): Json<SubmitTask>,
) -> StatusCode {
    tracing::info!("Creating a new task");
    if let Ok(task) = Task::try_from(submit) {
        state.append(Box::new(task));
        StatusCode::CREATED
    } else {
        StatusCode::BAD_REQUEST
    }
}

pub async fn delete_task(
    State(state): State<Queue>,
    path: axum::extract::Path<TaskID>,
) -> StatusCode {
    tracing::info!("Deleting task {}", path.0);
    if let Err(_) = state.delete(path.0) {
        StatusCode::NOT_FOUND
    } else {
        StatusCode::OK
    }
}

pub async fn clear_tasks(State(state): State<Queue>) -> StatusCode {
    tracing::info!("Clearing all tasks");
    if let Err(_) = state.clear() {
        StatusCode::INTERNAL_SERVER_ERROR
    } else {
        StatusCode::OK
    }
}

pub async fn process_tasks(State(state): State<Queue>) -> StatusCode {
    tracing::info!("Processing tasks");
    if let Err(_) = state.process() {
        StatusCode::INTERNAL_SERVER_ERROR
    } else {
        StatusCode::OK
    }
}