use glowrs::{Device, PoolingStrategy, SentenceTransformer};

use crate::server::data_models::{EmbeddingsRequest, EmbeddingsResponse};
use crate::server::infer::client::Client;
use crate::server::infer::handler::RequestHandler;
use crate::server::infer::DedicatedExecutor;

pub struct EmbeddingsHandler {
    sentence_transformer: SentenceTransformer,
}

impl EmbeddingsHandler {
    pub fn new(sentence_transformer: SentenceTransformer) -> Self {
        Self {
            sentence_transformer,
        }
    }
    pub fn from_repo_string(model_repo: &str, device: &Device) -> anyhow::Result<Self> {
        tracing::info!("Loading model: {}. Wait for model load.", model_repo);

        let sentence_transformer = SentenceTransformer::from_repo_string(model_repo, device)?;

        tracing::info!("Model loaded");

        Ok(Self {
            sentence_transformer,
        })
    }
}

impl RequestHandler for EmbeddingsHandler {
    type Input = EmbeddingsRequest;
    type Output = EmbeddingsResponse;

    fn handle(&mut self, request: EmbeddingsRequest) -> anyhow::Result<EmbeddingsResponse> {
        let sentences = request.input;

        // TODO: Is this even necessary?
        const NORMALIZE: bool = false;

        // Infer embeddings
        let (embeddings, usage) = self.sentence_transformer.encode_batch_with_usage(
            sentences.into(),
            NORMALIZE,
            PoolingStrategy::Mean,
        )?;

        let response = EmbeddingsResponse::from_embeddings(embeddings, usage, request.model);

        Ok(response)
    }
}

impl From<SentenceTransformer> for EmbeddingsHandler {
    fn from(sentence_transformer: SentenceTransformer) -> Self {
        Self::new(sentence_transformer)
    }
}

/// Embeddings inference struct
#[derive(Clone)]
pub struct EmbeddingsClient(Client<EmbeddingsHandler>);

impl EmbeddingsClient {
    pub(crate) fn new(executor: &DedicatedExecutor<EmbeddingsHandler>) -> Self {
        Self(Client::new(executor))
    }

    pub async fn generate_embedding(
        &self,
        request: EmbeddingsRequest,
    ) -> anyhow::Result<EmbeddingsResponse> {
        let rx = self.0.send(request).await?;
        rx.await
            .map_err(|_| anyhow::anyhow!("Failed to receive response from executor"))
    }
}
