use std::marker::PhantomData;
use anyhow::Result;
use tokio::sync::oneshot;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::time::Instant;
use uuid::Uuid;

use crate::server::data_models::{EmbeddingsRequest, EmbeddingsResponse};


// Generic types for task-specific data
type TaskId = Uuid;

// Marker trait for task requests
pub trait TaskRequest: Send + Sync + 'static {}

// Marker trait for task responses
pub trait TaskResponse: Send + Sync + 'static {}

impl TaskRequest for EmbeddingsRequest {}

impl TaskResponse for EmbeddingsResponse {}

// Trait representing a stateful task processor
pub trait TaskProcessor<T: TaskRequest, O: TaskResponse>: Send {
    fn new() -> Result<Self> where Self: Sized;
    fn handle_task(&mut self, request: T) -> Result<O>;
}

/// Queue entry
#[derive(Debug)]
pub(crate) struct QueueEntry<T: TaskRequest, O: TaskResponse> {
    /// Identifier
    pub id: TaskId,

    /// Request
    pub request: T,

    /// Response sender
    pub response_tx: oneshot::Sender<O>,

    /// Instant when this entry was queued
    pub queue_time: Instant,
}

impl<T: TaskRequest, O: TaskResponse> QueueEntry<T, O> {
    pub fn new(request: T, response_tx: oneshot::Sender<O>) -> Self {
        Self {
            id: Uuid::new_v4(),
            request,
            response_tx,
            queue_time: Instant::now(),
        }
    }
}

// Generic queue command for extensibility
#[derive(Debug)]
pub(crate) enum QueueCommand<T: TaskRequest, O: TaskResponse>
where T: Send
{
    Append(QueueEntry<T, O>),
    Stop,
}

/// Request Queue with stateful task processor
#[derive(Clone)]
pub struct Queue<TReq, TResp, Proc>
where TReq: TaskRequest,
      TResp: TaskResponse,
      Proc: TaskProcessor<TReq, TResp>
{
    tx: UnboundedSender<QueueCommand<TReq, TResp>>,
	_processor: PhantomData<Proc>
}

impl<TReq, TResp, Proc> Queue<TReq, TResp, Proc>
where TReq: TaskRequest,
      TResp: TaskResponse,
      Proc: TaskProcessor<TReq, TResp> + 'static
{
    pub(crate) fn new() -> Result<Self> {
        // TODO: Replace with MPMC w/ more worker threads
        // Create channel
        let (queue_tx, queue_rx) = unbounded_channel();

        // Create task processor
        let processor = Proc::new()?;

        std::thread::spawn(move || {
            // Create a new Runtime to run tasks
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("queue")
                .worker_threads(4)
                .build()?;

            // Pull task requests off the channel and send them to the executor
            runtime.block_on(queue_task(queue_rx, processor))
        });

        Ok(Self { tx: queue_tx, _processor: PhantomData::default() })
    }

    pub(crate) fn get_tx(&self) -> UnboundedSender<QueueCommand<TReq, TResp>> {
        self.tx.clone()
    }
}

// Generic background task executor with stateful processor
async fn queue_task<T: TaskRequest, O: TaskResponse, P: TaskProcessor<T, O>>(
    mut receiver: UnboundedReceiver<QueueCommand<T, O>>,
    mut processor: P,
) -> Result<()> {
    use QueueCommand::*;
    'main: while let Some(cmd) = receiver.recv().await {
        match cmd {
            Append(entry) => {
                // Process the task using the stateful processor
                let response = processor.handle_task(entry.request)?;
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



