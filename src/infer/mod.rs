pub mod queue;

use thiserror::Error;
use anyhow::Result;

use crate::infer::queue::{Entry, Queue};
use crate::server::data_models::{EmbeddingsRequest, EmbeddingsResponse};

#[derive(Debug, Error)]
pub enum InferError {
    #[error("Request failed during generation: {0}")]
    GenerationError(String),
}

/// Inference struct
#[derive(Clone)]
pub struct Infer {
	/// Request queue
	queue: Queue,
}

impl Infer {
	pub fn new() -> Result<Self> {
		let queue = Queue::new()?;

		Ok(Self { queue })
	}
	
	pub async fn generate_embedding(&self, request: EmbeddingsRequest) -> anyhow::Result<EmbeddingsResponse> {
		// Create channel for result communication
		let (queue_tx, queue_rx) = tokio::sync::oneshot::channel();

		// Create queue entry & append
		let entry = Entry::new(
            request,
            queue_tx,
        );
		self.queue.append(entry).await?;

		queue_rx.await?
	}
}
