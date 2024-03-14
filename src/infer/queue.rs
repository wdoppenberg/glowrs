use std::any::Any;
use std::collections::VecDeque;
use anyhow::Result;
use tokio::sync::oneshot;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::time::Instant;
use uuid::Uuid;

use crate::model::embedder::JinaBertModel;
use crate::model::sentence_transformer::SentenceTransformer;
use crate::server::data_models::{EmbeddingsRequest, EmbeddingsResponse};

// Generic types for task-specific data
type TaskId = Uuid;

// Marker trait for task requests
trait TaskRequest: Send + Sync + 'static {}

// Marker trait for task responses
trait TaskResponse: Send + Sync + 'static {}

// Trait representing a stateful task processor
trait TaskProcessor {
    // Concrete request and response types are defined within the implementation
    type Request: TaskRequest;
    type Response: TaskResponse;

    fn new() -> Result<Self> where Self: Sized;
    fn handle_task(&mut self, request: Self::Request) -> Result<Self::Response>;
}
/// Queue entry
#[derive(Debug)]
pub(crate) struct EmbeddingsEntry {
	/// Identifier
	pub id: TaskId,

    /// Request
    pub embeddings_request: EmbeddingsRequest,

    /// Response sender to communicate between the Infer struct and the batching_task
    pub response_tx: oneshot::Sender<EmbeddingsResponse>,

    /// Instant when this entry was queued
    pub queue_time: Instant,
}

impl EmbeddingsEntry {
    pub fn new(embeddings_request: EmbeddingsRequest, response_tx: oneshot::Sender<EmbeddingsResponse>) -> Self {
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
	mut receiver: UnboundedReceiver<QueueCommand<EmbeddingsEntry>>
) -> Result<()>{
	tracing::info!("Loading model: {}. Wait for model load.", args.model_repo);
    let sentence_transformer: SentenceTransformer<Embedder> = SentenceTransformer::from_repo(
	    args.model_repo, args.revision
    )?;
    tracing::info!("Model loaded");

	use QueueCommand::*;
    'main: while let Some(cmd) = receiver.recv().await {
        match cmd {
            Append(entry) => {
	            tracing::trace!("Processing entry {}, added {:?}s ago", entry.id, entry.queue_time.elapsed().as_secs());
                let sentences = entry.embeddings_request.input;

            	let normalize = true;
	            
	            // Infer embeddings
            	let (embeddings, usage) = sentence_transformer.encode_batch_with_usage(sentences, normalize)?;

            	let response = EmbeddingsResponse::from_embeddings(
		            embeddings, usage, entry.embeddings_request.model
	            );

	            // TODO: Handle result type
            	let _ = entry.response_tx.send(response);
            },
	        Stop => {
		        tracing::info!("Stopping queue task");
		        break 'main
	        }
        }
    }
	
	Ok(())
}

struct EmbeddingsTaskProcessor {
	sentence_transformer: SentenceTransformer<Embedder>
}



#[derive(Debug)]
pub(crate) enum QueueCommand<T>
where T: Send
{
    Append(T),
	Stop
}


/// Request Queue
#[derive(Clone)]
pub struct Queue<T> {
    tx: UnboundedSender<T>,
}

impl Queue<QueueCommand<EmbeddingsEntry>> {
    pub(crate) fn new() -> Result<Self> {
	    
        // Create channel
        let (queue_tx, queue_rx) = unbounded_channel();

	    // Inference args
	    let args = InferenceArgs {
		    model_repo: "jinaai/jina-embeddings-v2-base-en",
		    revision: "main"
	    };
	    
	    std::thread::spawn(move || {
            // Create a new Runtime to run tasks
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("queue")
                .worker_threads(4)
                // Lower OS priority of worker threads to prioritize main runtime
                // .on_thread_start(move || set_current_thread_priority_low())
                .build()?;

            // Pull task requests off the channel and send them to the executor
            runtime.block_on(queue_task(args, queue_rx))
        });

	    Ok(Self { tx: queue_tx })
    }
	
	pub(crate) fn get_tx(&self) -> UnboundedSender<QueueCommand<EmbeddingsEntry>> {
		self.tx.clone()
	}
}
