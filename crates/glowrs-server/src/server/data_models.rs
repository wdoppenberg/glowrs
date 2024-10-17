use candle_core::Tensor;
use glowrs::Usage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    Float,
    Base64,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct EmbeddingsRequest {
    pub input: Sentences,
    pub model: String,
    pub encoding_format: Option<EncodingFormat>,
    pub dimensions: Option<usize>,
    pub user: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<InnerEmbeddingsResponse>,
    pub model: String,
    pub usage: Usage,
}

impl EmbeddingsResponse {
    pub fn from_embeddings(embeddings: Tensor, usage: Usage, model: String) -> Self {
        let inner_responses: Vec<InnerEmbeddingsResponse> = embeddings
            .to_vec2()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| InnerEmbeddingsResponse {
                object: "core".to_string(),
                embedding,
                index: index as u32,
            })
            .collect();

        EmbeddingsResponse {
            object: "list".to_string(),
            data: inner_responses,
            model,
            usage,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct InnerEmbeddingsResponse {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: u32,
}

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
