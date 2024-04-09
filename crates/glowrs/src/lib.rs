#![doc = include_str!("../README.md")]

pub mod model;

pub use model::sentence_transformer::SentenceTransformer;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum Sentences {
    Single(String),
    Multiple(Vec<String>),
}

impl From<String> for Sentences {
    fn from(s: String) -> Self {
        Self::Single(s)
    }
}

impl From<Vec<&str>> for Sentences {
    fn from(strings: Vec<&str>) -> Self {
        Self::Multiple(strings.into_iter().map(|s| s.to_string()).collect())
    }
}

impl From<Vec<String>> for Sentences {
    fn from(strings: Vec<String>) -> Self {
        Self::Multiple(strings)
    }
}

impl From<Sentences> for Vec<String> {
    fn from(sentences: Sentences) -> Self {
        match sentences {
            Sentences::Single(s) => vec![s],
            Sentences::Multiple(vec) => vec,
        }
    }
}

#[derive(Debug, Serialize, PartialEq, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

