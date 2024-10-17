use std::marker::PhantomData;

/// Trait representing a (stateful) task processor that should run inside its
/// own thread.
pub trait RequestHandler
where
    Self: Send + Sized + 'static,
{
    type Input: Send + Sync + 'static;
    type Output: Send + Sync + 'static;

    fn handle(&mut self, request: Self::Input) -> anyhow::Result<Self::Output>;
}

pub struct CustomFnRequestHandler<F, Input, Output>
where
    Self: Send + 'static,
    F: Fn(Input) -> Output,
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    op: F,
    _req: PhantomData<Input>,
    _resp: PhantomData<Output>,
}

impl<F, Input, Output> CustomFnRequestHandler<F, Input, Output>
where
    Self: Send + 'static,
    F: Fn(Input) -> Output,
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    pub(crate) fn new(op: F) -> Self {
        Self {
            op,
            _req: PhantomData,
            _resp: PhantomData,
        }
    }
}

impl<F, Input, Output> From<F> for CustomFnRequestHandler<F, Input, Output>
where
    Self: Send + 'static,
    F: Fn(Input) -> Output,
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    fn from(op: F) -> Self {
        Self::new(op)
    }
}

impl<F, Input, Output> RequestHandler for CustomFnRequestHandler<F, Input, Output>
where
    Self: Send + 'static,
    F: Fn(Input) -> Output,
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    type Input = Input;
    type Output = Output;

    fn handle(&mut self, request: Self::Input) -> anyhow::Result<Self::Output> {
        Ok((self.op)(request))
    }
}

#[cfg(test)]
mod test {
    use candle_core::Tensor;
    use glowrs::core::device::DEVICE;

    use crate::server::infer::client::Client;
    use crate::server::infer::handler::CustomFnRequestHandler;
    use crate::server::infer::DedicatedExecutor;

    fn append_str(s_in: String) -> String {
        format!("{}-processed", s_in)
    }
    #[tokio::test]
    async fn test_from_simple() {
        let append_handler = CustomFnRequestHandler::from(append_str);

        let executor = DedicatedExecutor::new(append_handler).unwrap();

        let task = String::from("task");

        let client = Client::new(&executor);
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

        let t1 = Tensor::randn::<_, f32>(0., 2., (1, TENSOR_DIM), &DEVICE).unwrap();

        let tensor_handler = CustomFnRequestHandler::from(move |t2| some_tensor_op(&t1, &t2));

        let executor = DedicatedExecutor::new(tensor_handler).unwrap();

        let client = Client::new(&executor);

        let task = Tensor::randn::<_, f32>(0., 2., (TENSOR_DIM, 1), &DEVICE).unwrap();
        let rx = client.send(task).await.unwrap();

        let response = rx.await.unwrap();

        assert_eq!(response.dims()[0], 1);
        assert_eq!(response.dims()[1], 1);
    }
}
