
use anyhow::Result;
use tokio::sync::oneshot;
use tokio::sync::mpsc::UnboundedSender;

use crate::infer::queue::QueueCommand;
use crate::infer::{TaskRequest, TaskResponse};

pub(crate) trait Client {
    /// Type to send over channel / interface
    type SendType: TaskRequest;

    /// Type to receive over channel / interface
    type RecvType: TaskResponse;

	#[allow(async_fn_in_trait)]
    async fn send(
        &self,
        value: Self::SendType,
    ) -> Result<oneshot::Receiver<Self::RecvType>>;

	fn get_tx(&self) -> UnboundedSender<QueueCommand<Self::SendType, Self::RecvType>>;
}

