use anyhow::Result;
use crate::infer::embed::EmbeddingsProcessor;

use crate::infer::Queue;
use crate::infer::embed::EmbeddingsClient;
use crate::server::data_models::{EmbeddingsRequest, EmbeddingsResponse};

/// Represents the state of the server.
#[derive(Clone)]
pub struct ServerState {
    pub embeddings_client: EmbeddingsClient,
}

impl ServerState {
    pub fn new(
        embed_queue: &Queue<EmbeddingsRequest, EmbeddingsResponse, EmbeddingsProcessor>,
    ) -> Result<Self> {
        let embeddings_client = EmbeddingsClient::new(embed_queue.get_tx());

        Ok(Self { embeddings_client })
    }
}
