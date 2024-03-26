use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot;
use candle_transformers::models::jina_bert::BertModel as JinaBertModel;

use crate::infer::client::Client;
use crate::infer::Queue;
use crate::infer::queue::{QueueCommand, QueueEntry, RequestHandler};
use crate::model::sentence_transformer::SentenceTransformer;
use crate::server::data_models::{EmbeddingsRequest, EmbeddingsResponse};

/// Embeddings inference struct
#[derive(Clone)]
pub struct EmbeddingsClient {
    /// Queue sender
    tx: UnboundedSender<QueueCommand<EmbeddingsRequest, EmbeddingsResponse>>,
}

impl Client for EmbeddingsClient {
    type SendType = EmbeddingsRequest;
    type RecvType = EmbeddingsResponse;

    async fn send(
        &self,
        value: Self::SendType,
    ) -> anyhow::Result<oneshot::Receiver<Self::RecvType>> {
        let (queue_tx, queue_rx) = oneshot::channel();
        let entry = QueueEntry::new(value, queue_tx);
	    let command = QueueCommand::Append(entry);
        self.tx.send(command)?;

	    Ok(queue_rx)
    }

	fn get_tx(&self) -> UnboundedSender<QueueCommand<Self::SendType, Self::RecvType>> {
		self.tx.clone()
	}
}

impl EmbeddingsClient {
	pub(crate) fn new(queue: &Queue<EmbeddingsRequest, EmbeddingsResponse, EmbeddingsHandler>) -> Self {
		Self {
			tx: queue.tx.clone()
		}
	}
	pub async fn generate_embedding(&self, request: EmbeddingsRequest) -> anyhow::Result<EmbeddingsResponse> {
    	let rx = self.send(request).await?;
    	rx.await.map_err(|_| anyhow::anyhow!("Failed to receive response from queue"))
	}
}

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
	    // TODO: Don't hardcode
        tracing::info!("Loading model: {}. Wait for model load.", model_repo);
        let sentence_transformer: SentenceTransformer<Embedder> =
            SentenceTransformer::from_repo(model_repo, revision)?;
        tracing::info!("Model loaded");

        Ok(Self {
            sentence_transformer,
        })
    }
}

impl RequestHandler<EmbeddingsRequest, EmbeddingsResponse> for EmbeddingsHandler {
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
