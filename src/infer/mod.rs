pub mod queue;
mod client;

use thiserror::Error;

pub use client::{EmbeddingsClient};
pub use queue::Queue;

#[derive(Debug, Error)]
pub enum InferError {
    #[error("Request failed during generation: {0}")]
    GenerationError(String),
}

