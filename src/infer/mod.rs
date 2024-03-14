pub mod queue;
mod client;

use thiserror::Error;

pub use client::{Client, EmbeddingsClient};
pub use queue::Queue;

#[derive(Debug, Error)]
pub enum InferError {
    #[error("Request failed during generation: {0}")]
    GenerationError(String),
}

