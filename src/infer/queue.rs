use anyhow::Result;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use candle_transformers::models::bert::BertModel;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tokio::task::spawn_blocking;
use tokio::time::Instant;
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
            embeddings_request,
            response_tx,
            queue_time: Instant::now()
        }
    }
}

/// Queue State
#[derive(Debug)]
struct State {
    /// Queue entries organized in a Vec
    entries: VecDeque<(u64, Entry)>,

    /// Id of the next entry
    next_id: u64,

    /// Id of the next batch
    next_batch_id: u64,
}

impl State {
    fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            next_id: 0,
            next_batch_id: 0,
        }
    }

    /// Append an entry to the queue
    fn append(&mut self, entry: Entry) {
        // Push entry in the queue
        self.entries.push_back((self.next_id, entry));
        self.next_id += 1;
    }
}

// TODO: Configurable by feature flag?
type Embedder = BertModel;


// Background task responsible for the queue state
async fn queue_task(_state: Arc<Mutex<State>>, mut receiver: mpsc::UnboundedReceiver<QueueCommand>) {
    // TODO: Make configurable
    let model_repo = "sentence-transformers/all-MiniLM-L6-v2";
    let revision = "refs/pr/21".to_string();
    let sentence_transformer: SentenceTransformer<Embedder> = SentenceTransformer::from_repo(model_repo, revision).expect("Failed to load model.");
    tracing::info!("Model {} loaded", model_repo);

    while let Some(cmd) = receiver.recv().await {
        match cmd {
            // TODO: Replace with separate (blocking) inference task
            Append(entry) => {
                // state.lock().unwrap().append(*entry);
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


impl Default for State {
    fn default() -> Self {
        Self::new(128)
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
        // Create state
        let state = Arc::new(Mutex::new(State::default()));
        // Create channel
        let (queue_sender, queue_receiver) = mpsc::unbounded_channel();

        // Launch blocking background queue task
        // TODO: Not blocking atm, fix
        tokio::spawn(queue_task(state, queue_receiver));

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

