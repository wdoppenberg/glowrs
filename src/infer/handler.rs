use std::marker::PhantomData;

/// Trait representing a (stateful) task processor that should run inside its
/// own thread. 
pub trait RequestHandler
where
    Self: Send + Sized + 'static,
{
    type TReq: Send + Sync + 'static;
    type TResp: Send + Sync + 'static;
    
    fn handle(&mut self, request: Self::TReq) -> anyhow::Result<Self::TResp>;
}

pub struct CustomFnRequestHandler<F, TReq, TResp>
where
    Self: Send + 'static,
    F: Fn(TReq) -> TResp,
    TReq: Send + Sync + 'static,
    TResp: Send + Sync + 'static
{
    op: F,
    _req: PhantomData<TReq>,
    _resp: PhantomData<TResp>
}

impl<F, TReq, TResp> CustomFnRequestHandler<F, TReq, TResp>
where
    Self: Send + 'static,
    F: Fn(TReq) -> TResp,
    TReq: Send + Sync + 'static,
    TResp: Send + Sync + 'static
{
    pub(crate) fn new(op: F) -> Self {
        Self {op, _req: PhantomData, _resp: PhantomData}
    }
}

impl<F, TReq, TResp> From<F> for CustomFnRequestHandler<F, TReq, TResp>
where
    Self: Send + 'static,
    F: Fn(TReq) -> TResp,
    TReq: Send + Sync + 'static,
    TResp: Send + Sync + 'static
{
    fn from(op: F) -> Self {
        Self::new(op)
    }
}

impl<F, TReq, TResp> RequestHandler for CustomFnRequestHandler<F, TReq, TResp>
where
    Self: Send + 'static,
    F: Fn(TReq) -> TResp,
    TReq: Send + Sync + 'static,
    TResp: Send + Sync + 'static
{
    type TReq = TReq;
    type TResp = TResp;

    fn handle(&mut self, request: Self::TReq) -> anyhow::Result<Self::TResp> {
        Ok((self.op)(request))
    }
}

#[cfg(test)]
mod test {
    use candle_core::Tensor;

    use super::*;
    use crate::infer::client::Client;
    use crate::infer::Queue;
    use crate::model::device::DEVICE;

    fn append_str(s_in: String) -> String {
        format!("{}-processed", s_in)
    }
    #[tokio::test]
    async fn test_from_simple() {
        let append_handler = CustomFnRequestHandler::from(append_str);

        let queue = Queue::new(append_handler).unwrap();

        let task = String::from("task");

        let client = Client::new(&queue);
        let rx = client.send(task).await.unwrap();

        let response = rx.await.unwrap();

        assert_eq!("task-processed", response);
    }

    fn some_tensor_op(t1: &Tensor, t2: &Tensor) -> Tensor {
        t1.matmul(t2).unwrap()
    }

    #[tokio::test]
    async fn test_with_move() {
        const TENSOR_DIM: usize = 512;

        let t1 = Tensor::randn(0., 2., (1, TENSOR_DIM), &DEVICE).unwrap();

        let tensor_handler = CustomFnRequestHandler::from(
            move |t2| some_tensor_op(&t1, &t2)
        );

        let queue = Queue::new(tensor_handler).unwrap();

        let client = Client::new(&queue);

        let task = Tensor::randn(0., 2., (TENSOR_DIM, 1), &DEVICE).unwrap();
        let rx = client.send(task).await.unwrap();

        let response = rx.await.unwrap();

        assert_eq!(response.dims()[0], 1);
        assert_eq!(response.dims()[1], 1);
    }
}