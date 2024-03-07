use anyhow::Result;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use uuid::Uuid;
use crate::embedding::embedder::JinaBertModel;
use crate::embedding::sentence_transformer::SentenceTransformer;

use crate::server::routes::{EmbeddingsRequest, EmbeddingsResponse, InnerEmbeddingsResponse};
use crate::infer::queue::QueueCommand::Append;

#[derive(Debug, Error)]
pub enum InferError {
    #[error("Request failed during generation: {0}")]
    GenerationError(String),
}

/// Queue entry
#[derive(Debug)]
pub(crate) struct Entry {
	/// Identifier
	pub id: Uuid,

    /// Request
    pub embeddings_request: EmbeddingsRequest,

    /// Response sender to communicate between the Infer struct and the batching_task
    pub response_tx: oneshot::Sender<Result<EmbeddingsResponse>>,

    /// Instant when this entry was queued
    pub queue_time: Instant,
}

impl Entry {
    pub fn new(embeddings_request: EmbeddingsRequest, response_tx: oneshot::Sender<Result<EmbeddingsResponse>>) -> Self {
        Self {
	        id: Uuid::new_v4(),
            embeddings_request,
            response_tx,
            queue_time: Instant::now()
        }
    }
}


// TODO: Configurable by feature flag?
type Embedder = JinaBertModel;

struct InferenceArgs {
	model_repo: &'static str,
	revision: &'static str
}

// Background task responsible for the queue state
async fn queue_task(
	args: InferenceArgs,
	mut receiver: mpsc::UnboundedReceiver<QueueCommand>
) {
	tracing::info!("Loading model: {}. Wait for model load.", args.model_repo);
    let sentence_transformer: SentenceTransformer<Embedder> = SentenceTransformer::from_repo(
	    args.model_repo, args.revision
    ).expect("Failed to load model.");
    tracing::info!("Model loaded");

    while let Some(cmd) = receiver.recv().await {
        match cmd {
            // TODO: Replace with separate (blocking) inference task
            Append(entry) => {
	            tracing::trace!("Processing entry {}, added {:?}", entry.id, entry.queue_time);
                let sentences = entry.embeddings_request.input;

            	let normalize = true;
            	let (embeddings, usage) = sentence_transformer.encode_batch_with_usage(sentences, normalize).unwrap();
            	let inner_responses: Vec<InnerEmbeddingsResponse> = embeddings
            		.to_vec2().unwrap()
                    .into_iter()
            		.enumerate()
            		.map(|(index, embedding)| InnerEmbeddingsResponse {
            			object: "embedding".to_string(),
            			embedding,
            			index: index as u32,
            		})
            		.collect();

            	let response = EmbeddingsResponse {
            		object: "list".to_string(),
            		data: inner_responses,
            		model: entry.embeddings_request.model,
            		usage,
            	};

            	let _ = entry.response_tx.send(Ok(response));
            }
        }
    }
}

#[derive(Debug)]
enum QueueCommand {
    Append(Box<Entry>),
}

/// Request Queue
#[derive(Debug, Clone)]
pub(crate) struct Queue {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommand>,
}

impl Queue {
    pub(crate) fn new() -> Self {
        // Create channel
        let (queue_sender, queue_receiver) = mpsc::unbounded_channel();

	    // Inference args
	    let args = InferenceArgs {
		    model_repo: "jinaai/jina-embeddings-v2-base-en",
		    revision: "main"
	    };

        // Launch blocking background queue task
        // TODO: Not blocking atm, investigate whether necessary
        tokio::spawn(queue_task(args, queue_receiver));

        Self { queue_sender }
    }

    pub(crate) async fn append(
        &self,
        entry: Entry,
    ) -> Result<()> {
        let _ = self.queue_sender.send(Append(Box::new(entry)));

        Ok(())
    }
}

