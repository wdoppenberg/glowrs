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
