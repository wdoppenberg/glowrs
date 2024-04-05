use tokio::sync::oneshot;
use uuid::Uuid;
use tokio::time::Instant;

use crate::infer::handler::RequestHandler;
use crate::infer::TaskId;

/// Queue entry
#[derive(Debug)]
pub(crate) struct QueueEntry<THandler>
where
    THandler: RequestHandler
{
    /// Identifier
    pub id: TaskId,

    /// Request
    pub request: THandler::TReq,

    /// Response sender
    pub response_tx: oneshot::Sender<THandler::TResp>,

    /// Instant when this entry was queued
    pub queue_time: Instant,
}

impl<THandler> QueueEntry<THandler> 
where THandler: RequestHandler 
{
    pub fn new(request: THandler::TReq, response_tx: oneshot::Sender<THandler::TResp>) -> Self {
        Self {
            id: Uuid::new_v4(),
            request,
            response_tx,
            queue_time: Instant::now(),
        }
    }
}
