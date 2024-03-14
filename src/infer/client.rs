
use anyhow::Result;
use tokio::sync::oneshot;
use tokio::sync::mpsc::UnboundedSender;

use crate::infer::queue::{EmbeddingsEntry, QueueCommand};
use crate::server::data_models::{EmbeddingsRequest, EmbeddingsResponse};


/// Marker trait to identify data being sent/received
pub trait Message {}



pub trait CommunicationChannel<T: Message> {
    async fn send(&mut self, msg: T) -> Result<()>;
    async fn recv(&mut self) -> Result<T>;
}

pub trait Client {
    /// Type to send over channel / interface
    type SendType;

    /// Type to receive over channel / interface
    type RecvType;
	
	#[allow(async_fn_in_trait)]
    async fn send(
        &self,
        value: Self::SendType,
    ) -> Result<oneshot::Receiver<Self::RecvType>>;
}

/// Embeddings inference struct
#[derive(Clone)]
pub struct EmbeddingsClient {
    /// Queue sender
    tx: UnboundedSender<QueueCommand<EmbeddingsEntry>>,
}

impl Client for EmbeddingsClient {
    type SendType = EmbeddingsRequest;
    type RecvType = EmbeddingsResponse;

    async fn send(
        &self,
        value: Self::SendType,
    ) -> Result<oneshot::Receiver<Self::RecvType>> {
        let (queue_tx, queue_rx) = oneshot::channel();
        let entry = EmbeddingsEntry::new(value, queue_tx);
	    let command = QueueCommand::Append(entry);
        self.tx.send(command)?;

	    Ok(queue_rx)
    }
}

impl Drop for EmbeddingsClient {
	fn drop(&mut self) {
		self.tx.send(QueueCommand::Stop).expect("Couldn't stop worker thread.");
	}
}

impl EmbeddingsClient {
	pub fn new(tx: UnboundedSender<QueueCommand<EmbeddingsEntry>>) -> Self {
		Self {
			tx
		}
	}
	pub async fn generate_embedding(&self, request: EmbeddingsRequest) -> Result<EmbeddingsResponse> {
    	let rx = self.send(request).await?;
    	rx.await.map_err(|_| anyhow::anyhow!("Failed to receive response from queue"))
	}
}

