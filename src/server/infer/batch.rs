use tokio::sync::oneshot;
use uuid::Uuid;
use tokio::time::Instant;

use crate::server::infer::handler::RequestHandler;
use crate::server::infer::TaskId;

/// Queue entry
#[derive(Debug)]
pub(crate) struct QueueEntry<THandler>
where
    THandler: RequestHandler
{
    /// Identifier
    pub id: TaskId,

    /// Request
    pub request: THandler::Input,

    /// Response sender
    pub response_tx: oneshot::Sender<THandler::Output>,

    /// Instant when this entry was queued
    pub queue_time: Instant,
}

impl<THandler> QueueEntry<THandler> 
where THandler: RequestHandler 
{
    pub fn new(request: THandler::Input, response_tx: oneshot::Sender<THandler::Output>) -> Self {
        Self {
            id: Uuid::new_v4(),
            request,
            response_tx,
            queue_time: Instant::now(),
        }
    }
}
