#![feature(associated_type_defaults)]

use std::fmt::Debug;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Deserialize, Serialize)]
pub struct SubmitTask {
    pub(crate) description: String,
}
pub type TaskID = Uuid;

pub trait Task: Debug + Clone + Send {
    type Input;
    type Output;

    fn from_input(input: Self::Input) -> Self;

    fn process(&self) -> Self::Output;

    fn get_id(&self) -> TaskID;
}