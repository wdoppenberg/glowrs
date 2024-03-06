use anyhow::Result;
use tokio::sync::oneshot;

use crate::{infer::queue::Queue, server::routes::EmbeddingsRequest};
use crate::infer::queue::Entry;
use crate::server::routes::EmbeddingsResponse;

/// Inference struct
#[derive(Clone)]
pub struct Infer {
	/// Request queue
	queue: Queue,
}

impl Infer {
	/// Create a new inference struct
	pub fn new() -> Self {
		let queue = Queue::new();

		Self { queue }
	}

	pub async fn generate_embedding(&self, request: EmbeddingsRequest) -> Result<EmbeddingsResponse> {
		// Create channel for result communication
		let (queue_tx, queue_rx) = oneshot::channel();

		// Create queue entry & append
		let entry = Entry::new(
            request,
            queue_tx,
        );
		self.queue.append(entry).await?;

		queue_rx.await?
	}
}