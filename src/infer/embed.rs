use candle_transformers::models::jina_bert::BertModel as JinaBertModel;

use crate::infer::client::Client;
use crate::infer::handler::RequestHandler;
use crate::infer::Queue;
use crate::model::sentence_transformer::SentenceTransformer;
use crate::server::data_models::{EmbeddingsRequest, EmbeddingsResponse};

type Embedder = JinaBertModel;

pub struct EmbeddingsHandler {
    sentence_transformer: SentenceTransformer<Embedder>,
}


impl EmbeddingsHandler {
	pub fn new(
		model_repo: &str,
		revision: &str,
	) -> anyhow::Result<Self>
    {
        tracing::info!("Loading model: {}. Wait for model load.", model_repo);
        let sentence_transformer: SentenceTransformer<Embedder> =
            SentenceTransformer::from_repo(model_repo, revision)?;
        tracing::info!("Model loaded");

        Ok(Self {
            sentence_transformer,
        })
    }
}

impl RequestHandler for EmbeddingsHandler {
	type TReq = EmbeddingsRequest;
	type TResp = EmbeddingsResponse;


	fn handle(&mut self, request: EmbeddingsRequest) -> anyhow::Result<EmbeddingsResponse> {
        let sentences = request.input;

	    // TODO: Is this even necessary?
        const NORMALIZE: bool = false;

        // Infer embeddings
        let (embeddings, usage) = self
            .sentence_transformer
            .encode_batch_with_usage(sentences, NORMALIZE)?;

        let response = EmbeddingsResponse::from_embeddings(embeddings, usage, request.model);

        Ok(response)
    }
}

/// Embeddings inference struct
#[derive(Clone)]
pub struct EmbeddingsClient(Client<EmbeddingsHandler>);


impl EmbeddingsClient {
	pub(crate) fn new(queue: &Queue<EmbeddingsHandler>) -> Self {
		Self(Client::new(queue))
	}
	pub async fn generate_embedding(&self, request: EmbeddingsRequest) -> anyhow::Result<EmbeddingsResponse> {
		let rx = self.0.send(request).await?;
		rx.await.map_err(|_| anyhow::anyhow!("Failed to receive response from queue"))
	}
}