use std::sync::Arc;
use axum::{debug_handler, http, Json};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use axum::extract::State;
use axum::http::StatusCode;
use crate::embedding::{Embedder, SentenceTransformer};

pub struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Something went wrong: {}", self.0),
        )
            .into_response()
    }
}


fn mock_response(model: &str) -> Result<EmbeddingsResponse, AppError> {
    Ok(EmbeddingsResponse {
        object: "list".into(),
        data: vec![InnerEmbeddingsResponse {
            object: "embedding".into(),
            embedding: vec![0.0023064255, -0.009327292, -0.0028842222],
            index: 0,
        }],
        model: model.to_string(),
        usage: Usage {
            prompt_tokens: 8,
            total_tokens: 8,
        },
    })
}

pub async fn text_embeddings(State(embedder): State<Arc<Embedder>>, Json(payload): Json<EmbeddingsRequest>) -> Result<impl IntoResponse, AppError> {
    let sentences = match payload.input {
        Sentences::Single(s) => vec![s],
        Sentences::Multiple(vec) => vec
    };

    let sentences = sentences.iter().map(|s| s.as_str()).collect();
    let embeddings = embedder.encode_batch(sentences, true).unwrap();

    let data = embeddings.to_vec2::<f32>().unwrap();

    let response = EmbeddingsResponse {
        object: "list".into(),
        data: data.into_iter().map(|vec| InnerEmbeddingsResponse {
                object: "embedding".into(),
                embedding: vec,
                index: 0,
            }).collect(),
        model: payload.model.to_string(),
        usage: Usage {
            prompt_tokens: 8,
            total_tokens: 8,
        },
    };

    Ok((http::StatusCode::OK, Json(response)))
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Sentences {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub input: Sentences,
    pub model: String,
    pub encoding_format: String,
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
            usage: Default::default()
        }
    }
}

#[derive(Debug, Serialize)]
pub struct InnerEmbeddingsResponse {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: u32,
}

#[derive(Debug, Serialize, PartialEq, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}
