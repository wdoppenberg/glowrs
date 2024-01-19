use std::fmt::Debug;
use uuid::Uuid;

pub type TaskID = Uuid;

pub trait Task: Debug + Clone + Send + 'static {
    type Input: Send;
    type Output: Send;
    fn process(&self) -> Self::Output;

    fn get_id(&self) -> TaskID;
}
