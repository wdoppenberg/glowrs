use crate::embedding::embedder::Embedder;
use crate::embedding::sentence_transformer::SentenceTransformer;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

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

pub async fn text_embeddings<M>(
    State(embedder_mutex): State<Arc<Mutex<SentenceTransformer<M>>>>,
    Json(payload): Json<EmbeddingsRequest>,
) -> Result<impl IntoResponse, TextEmbeddingRouteError>
where
    M: Embedder,
{
    // // Extracting the model version from the model repo name
    // let model_version: String = match String::From(M).rsplit('/').next() {
    //     Some(version) => version.to_string(),
    //     None => M::MODEL_REPO_NAME.to_string(),
    // };
    //
    // if payload.model != model_version {
    //     return Err(TextEmbeddingRouteError::from(anyhow::anyhow!(
    //         "Model version mismatch"
    //     )));
    // }
    let sentences: Sentences = payload.input;

    let embedder = embedder_mutex.lock().await;
    let (embeddings, usage) = embedder.encode_batch_with_usage(sentences, true)?;

    let data = embeddings.to_vec2::<f32>()?;

    let response = EmbeddingsResponse {
        object: "list".into(),
        data: data
            .into_iter()
            .enumerate()
            .map(|(i, vec)| InnerEmbeddingsResponse {
                object: "embedding".into(),
                embedding: vec,
                index: i as u32,
            })
            .collect(),
        model: payload.model,
        usage,
    };

    Ok((StatusCode::OK, Json(response)))
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
pub struct EmbeddingsRequest {
    pub input: Sentences,
    pub model: String,
    pub encoding_format: String,
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
