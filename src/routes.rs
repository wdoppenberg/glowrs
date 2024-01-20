use std::sync::Arc;
use axum::Json;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use axum::extract::State;
use axum::http::StatusCode;
use crate::embedding::models::SBert;
use crate::embedding::sentence_transformer::{SentenceTransformer, Usage};

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

pub async fn text_embeddings<M>(State(embedder): State<Arc<SentenceTransformer<M>>>, Json(payload): Json<EmbeddingsRequest>)
                             -> Result<impl IntoResponse, TextEmbeddingRouteError>
where M: SBert
{
	    // Extracting the model version from the model repo name
    let model_version: String = match M::MODEL_REPO_NAME.rsplit('/').next() {
        Some(version) => version.to_string(),
        None => {
            M::MODEL_REPO_NAME.to_string()
        }
    };

    if payload.model != model_version {
	    return Err(TextEmbeddingRouteError::from(anyhow::anyhow!("Model version mismatch")));
    }

	let sentences: Vec<String> = payload.input.into();

	let sentences = sentences.iter().map(|s| s.as_str()).collect();
	let (embeddings, usage) = embedder.encode_batch_with_usage(sentences, true)?;

	let data = embeddings.to_vec2::<f32>()?;

	let response = EmbeddingsResponse {
		object: "list".into(),
		data: data.into_iter()
			.enumerate()
			.map(|(i, vec)| InnerEmbeddingsResponse {
			object: "embedding".into(),
			embedding: vec,
			index: i as u32,
		}).collect(),
		model: model_version,
		usage
	};

	Ok((StatusCode::OK, Json(response)))
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Sentences {
	Single(String),
	Multiple(Vec<String>),
}

impl From<Sentences> for Vec<String> {
    fn from(sentences: Sentences) -> Self {
        match sentences {
            Sentences::Single(s) => vec![s],
            Sentences::Multiple(vec) => vec
        }
    }
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

