use std::marker::PhantomData;
use anyhow::Result;
use tokio::sync::oneshot;
use tokio::sync::mpsc::UnboundedSender;

use crate::infer::handler::RequestHandler;
use crate::infer::Queue;
use crate::infer::queue::{QueueCommand, QueueEntry};


pub(crate) struct Client<THandler> 
where THandler: RequestHandler
{
	tx: UnboundedSender<QueueCommand<THandler::TReq, THandler::TResp>>,
	_phantom: PhantomData<THandler>
}

impl<THandler> Client<THandler> 
where THandler: RequestHandler
{
	pub(crate) fn new(queue: &Queue<THandler>) -> Self {
		Self {
			tx: queue.tx.clone(),
			_phantom: PhantomData
		}
	}
	
	pub(crate) async fn send(
		&self,
		value: THandler::TReq,
    ) -> Result<oneshot::Receiver<THandler::TResp>> {
		// Create channel
        let (queue_tx, queue_rx) = oneshot::channel();
		
		// Create command
        let entry = QueueEntry::new(value, queue_tx);
	    let command = QueueCommand::Append(entry);
		
		// Send command
        self.tx.send(command)?;
		
		// Return receiver
	    Ok(queue_rx)
	}
}

impl<THandler> Clone for Client<THandler> 
where THandler: RequestHandler
{
    fn clone(&self) -> Self {
        Client {
            tx: self.tx.clone(),
            _phantom: PhantomData {},
        }
    }
}

