use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot;
use candle_transformers::models::jina_bert::BertModel as JinaBertModel;

use crate::infer::client::Client;
use crate::infer::queue::{QueueCommand, QueueEntry, RequestHandler};
use crate::infer::{TaskRequest, TaskResponse};
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
	pub(crate) fn new(tx: UnboundedSender<QueueCommand<EmbeddingsRequest, EmbeddingsResponse>>) -> Self {
		Self {
			tx
		}
	}
	pub async fn generate_embedding(&self, request: EmbeddingsRequest) -> anyhow::Result<EmbeddingsResponse> {
    	let rx = self.send(request).await?;
    	rx.await.map_err(|_| anyhow::anyhow!("Failed to receive response from queue"))
	}
}

type Embedder = JinaBertModel;

pub struct EmbeddingsProcessor {
    sentence_transformer: SentenceTransformer<Embedder>,
}

impl RequestHandler<EmbeddingsRequest, EmbeddingsResponse> for EmbeddingsProcessor {
    fn new() -> anyhow::Result<Self>
    {
	    // TODO: Don't hardcode
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

    fn handle(&mut self, request: EmbeddingsRequest) -> anyhow::Result<EmbeddingsResponse> {
        let sentences = request.input;

	    // TODO: Don't hardcode
        let normalize = false;

        // Infer embeddings
        let (embeddings, usage) = self
            .sentence_transformer
            .encode_batch_with_usage(sentences, normalize)?;

        let response = EmbeddingsResponse::from_embeddings(embeddings, usage, request.model);

        Ok(response)
    }
}

impl TaskRequest for EmbeddingsRequest {}

impl TaskResponse for EmbeddingsResponse {}
