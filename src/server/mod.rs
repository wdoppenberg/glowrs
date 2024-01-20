use serde::{Deserialize, Serialize};
use axum::extract::State;
use std::sync::Arc;
use axum::http::StatusCode;
use axum::Json;
use crate::work::queue::Queue;
use crate::work::Task;

#[derive(Deserialize, Serialize)]
pub struct SubmitTask {
    pub(crate) description: String,
}

pub async fn create_task<T: Task>(
    State(queue): State<Arc<Queue<T>>>,
    Json(submit): Json<T::Input>,
) -> StatusCode {
    tracing::info!("Creating a new task");
    unimplemented!()
}

pub async fn clear_tasks<T: Task>(State(state): State<Arc<Queue<T>>>) -> StatusCode {
    tracing::info!("Clearing all tasks");
    if state.clear().is_err() {
        StatusCode::INTERNAL_SERVER_ERROR
    } else {
        StatusCode::OK
    }
}

pub async fn process_tasks<T: Task>(State(state): State<Arc<Queue<T>>>) -> StatusCode {
    tracing::info!("Processing tasks");
    if state.process().is_err() {
        StatusCode::INTERNAL_SERVER_ERROR
    } else {
        StatusCode::OK
    }
}
