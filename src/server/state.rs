use anyhow::Result;

use crate::infer::{EmbeddingsClient, Queue};
use crate::infer::queue::{EmbeddingsEntry, QueueCommand};


/// Represents the state of the server.
#[derive(Clone)]
pub struct ServerState {
	pub embeddings_client: EmbeddingsClient,
}


impl ServerState {
	pub fn new(queue: &Queue<QueueCommand<EmbeddingsEntry>>) -> Result<Self> {
		let embeddings_client = EmbeddingsClient::new(queue.get_tx());
		
		Ok(Self { embeddings_client })
	}
}
