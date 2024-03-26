use anyhow::Result;
use crate::infer::embed::EmbeddingsHandler;

use crate::infer::Queue;
use crate::infer::embed::EmbeddingsClient;


/// Represents the state of the server.
#[derive(Clone)]
pub struct ServerState {
    pub embeddings_client: EmbeddingsClient,
    // pub embeddings_queue: Queue<EmbeddingsRequest, EmbeddingsResponse, EmbeddingsHandler>,
}

impl ServerState {
    pub fn new(
        embeddings_handler: EmbeddingsHandler,
    ) -> Result<Self> {
        let embeddings_queue = Queue::new(embeddings_handler)?;
        
        let embeddings_client = EmbeddingsClient::new(&embeddings_queue);

        Ok(Self { embeddings_client })
    }
}
