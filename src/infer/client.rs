
use anyhow::Result;
use tokio::sync::oneshot;
use tokio::sync::mpsc::UnboundedSender;

use crate::infer::queue::QueueCommand;

pub(crate) trait Client {
    /// Type to send over channel / interface
    type SendType: Send + Sync + 'static;

    /// Type to receive over channel / interface
    type RecvType: Send + Sync + 'static;

	#[allow(async_fn_in_trait)]
    async fn send(
        &self,
        value: Self::SendType,
    ) -> Result<oneshot::Receiver<Self::RecvType>>;

	// TODO: Get rid of this
	fn get_tx(&self) -> UnboundedSender<QueueCommand<Self::SendType, Self::RecvType>>;
}

