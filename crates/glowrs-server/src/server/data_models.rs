use candle_core::Tensor;
use glowrs::Sentences;
use glowrs::Usage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    Float,
    Base64,
}

#[derive(Debug, Deserialize, Clone)]
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
                object: "model".to_string(),
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
