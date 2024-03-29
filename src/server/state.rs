use std::sync::Arc;
use anyhow::Result;
use crate::infer::embed::EmbeddingsHandler;

use crate::infer::Queue;
use crate::infer::embed::EmbeddingsClient;


/// Represents the state of the server.
#[derive(Clone)]
pub struct ServerState {
    pub embeddings_client: EmbeddingsClient,
    // TODO: Fix queue + handler thread despawning
    pub embeddings_queue: Arc<Queue<EmbeddingsHandler>>,
}


impl ServerState {
    pub fn new(
        embeddings_handler: EmbeddingsHandler,
    ) -> Result<Self> {
        let embeddings_queue = Queue::new(embeddings_handler)?;

        let embeddings_client = EmbeddingsClient::new(&embeddings_queue);

        Ok(Self { embeddings_client, embeddings_queue: Arc::new(embeddings_queue) })
    }
}