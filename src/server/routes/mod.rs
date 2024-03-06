use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

pub mod default;
pub mod text_embeddings;

pub struct TextEmbeddingRouteError(anyhow::Error);

impl IntoResponse for TextEmbeddingRouteError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Something went wrong: {}", self.0),
        )
            .into_response()
    }
}

impl<E> From<E> for TextEmbeddingRouteError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    Float,
    Base64,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub input: Sentences,
    pub model: String,
    pub encoding_format: EncodingFormat,
}

#[derive(Debug, Serialize, PartialEq, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<InnerEmbeddingsResponse>,
    pub model: String,
    pub usage: Usage,
}

impl EmbeddingsResponse {
    pub fn empty() -> Self {
        Self {
            object: "list".into(),
            data: Vec::new(),
            model: "".into(),
            usage: Default::default(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct InnerEmbeddingsResponse {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: u32,
}
