#![doc = include_str!("../README.md")]

pub mod core;
mod error;
mod exports;

pub(crate) mod pooling;

pub use exports::*;

pub use crate::error::{Error, Result};

pub use core::sentence_transformer::SentenceTransformer;
pub use pooling::PoolingStrategy;

use serde::Serialize;

#[derive(Debug, Serialize, PartialEq, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}
