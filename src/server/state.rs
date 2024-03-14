use anyhow::Result;

use crate::infer::queue::TaskProcessor;
use crate::infer::{EmbeddingsClient, Queue};
use crate::model::embedder::JinaBertModel;
use crate::model::sentence_transformer::SentenceTransformer;
use crate::server::data_models::{EmbeddingsRequest, EmbeddingsResponse};

/// Represents the state of the server.
#[derive(Clone)]
pub struct ServerState {
    pub embeddings_client: EmbeddingsClient,
}

impl ServerState {
    pub fn new(
        queue: &Queue<EmbeddingsRequest, EmbeddingsResponse, EmbeddingsProcessor>,
    ) -> Result<Self> {
        let embeddings_client = EmbeddingsClient::new(queue.get_tx());

        Ok(Self { embeddings_client })
    }
}

type Embedder = JinaBertModel;

pub struct EmbeddingsProcessor {
    sentence_transformer: SentenceTransformer<Embedder>,
}

impl TaskProcessor<EmbeddingsRequest, EmbeddingsResponse> for EmbeddingsProcessor {
    fn new() -> Result<Self>
    {
        let model_repo = "jinaai/jina-embeddings-v2-base-en";
        let revision = "main";
        tracing::info!("Loading model: {}. Wait for model load.", model_repo);
        let sentence_transformer: SentenceTransformer<Embedder> =
            SentenceTransformer::from_repo(model_repo, revision)?;
        tracing::info!("Model loaded");

        Ok(Self {
            sentence_transformer,
        })
    }

    fn handle_task(&mut self, request: EmbeddingsRequest) -> Result<EmbeddingsResponse> {
        let sentences = request.input;

        let normalize = false;

        // Infer embeddings
        let (embeddings, usage) = self
            .sentence_transformer
            .encode_batch_with_usage(sentences, normalize)?;

        let response = EmbeddingsResponse::from_embeddings(embeddings, usage, request.model);

        Ok(response)
    }
}
