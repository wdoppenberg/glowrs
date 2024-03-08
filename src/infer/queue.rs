use std::sync::{Arc, Mutex};
use anyhow::Result;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use uuid::Uuid;
use crate::embedding::embedder::JinaBertModel;
use crate::embedding::sentence_transformer::SentenceTransformer;

use crate::infer::queue::QueueCommand::Append;
use crate::server::data_models::{EmbeddingsRequest, EmbeddingsResponse, InnerEmbeddingsResponse};

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
// TODO: Introduce custom error type
async fn queue_task(
	args: InferenceArgs,
	mut receiver: mpsc::UnboundedReceiver<QueueCommand>
) -> Result<()>{
	tracing::info!("Loading model: {}. Wait for model load.", args.model_repo);
    let sentence_transformer: SentenceTransformer<Embedder> = SentenceTransformer::from_repo(
	    args.model_repo, args.revision
    )?;
    tracing::info!("Model loaded");

    while let Some(cmd) = receiver.recv().await {
        match cmd {
            // TODO: Replace with separate (blocking) inference task
            Append(entry) => {
	            tracing::trace!("Processing entry {}, added {:?}s ago", entry.id, entry.queue_time.elapsed().as_secs());
                let sentences = entry.embeddings_request.input;

            	let normalize = true;
	            // Infer embeddings
            	let (embeddings, usage) = sentence_transformer.encode_batch_with_usage(sentences, normalize)?;

            	let response = EmbeddingsResponse::from_embeddings(
		            embeddings, usage, entry.embeddings_request.model
	            );

            	let _ = entry.response_tx.send(Ok(response));
            }
        }
    }
	
	Ok(())
}

#[derive(Debug)]
enum QueueCommand {
    Append(Box<Entry>),
}


struct State {
    /// Channel to communicate with the background queue task
	thread_join_handle: std::thread::JoinHandle<Result<()>>
}

/// Request Queue
#[derive(Clone)]
pub(crate) struct Queue {
    tx: mpsc::UnboundedSender<QueueCommand>,
	state: Arc<Mutex<State>>
}

impl Queue {
    pub(crate) fn new() -> Result<Self> {
	    
        // Create channel
        let (queue_tx, queue_rx) = mpsc::unbounded_channel();

	    // Inference args
	    let args = InferenceArgs {
		    model_repo: "jinaai/jina-embeddings-v2-base-en",
		    revision: "main"
	    };
	    
	    let thread = std::thread::spawn(move || {
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


        let state = State { thread_join_handle: thread };
	    
	    Ok(Self { tx: queue_tx, state: Arc::new(Mutex::new(state))})
    }

    pub(crate) async fn append(
        &self,
        entry: Entry,
    ) -> Result<()> {
	    
        self.tx.send(Append(Box::new(entry)))?;

        Ok(())
    }
}
