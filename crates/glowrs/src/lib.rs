#![doc = include_str!("../README.md")]

mod error;
pub mod model;
pub use error::Error;
pub(crate) use error::Result;

pub use model::pooling::PoolingStrategy;
pub use model::sentence_transformer::SentenceTransformer;
use serde::Serialize;

#[derive(Debug, Serialize, PartialEq, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}
