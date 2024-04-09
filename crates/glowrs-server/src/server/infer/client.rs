use anyhow::Result;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot;

use crate::server::infer::batch::QueueEntry;
use crate::server::infer::executor::Command;
use crate::server::infer::handler::RequestHandler;
use crate::server::infer::DedicatedExecutor;

pub(crate) struct Client<THandler>
where
    THandler: RequestHandler,
{
    tx: UnboundedSender<Command<THandler>>,
}

impl<THandler> Client<THandler>
where
    THandler: RequestHandler,
{
    pub(crate) fn new(executor: &DedicatedExecutor<THandler>) -> Self {
        Self {
            tx: executor.tx.clone(),
        }
    }

    pub(crate) async fn send(
        &self,
        value: THandler::Input,
    ) -> Result<oneshot::Receiver<THandler::Output>> {
        // Create channel
        let (tx, rx) = oneshot::channel();

        // Create command
        let entry = QueueEntry::new(value, tx);
        let command = Command::Append(entry);

        // Send command
        self.tx.send(command)?;

        // Return receiver
        Ok(rx)
    }
}

impl<THandler> Clone for Client<THandler>
where
    THandler: RequestHandler,
{
    fn clone(&self) -> Self {
        Client {
            tx: self.tx.clone(),
        }
    }
}
