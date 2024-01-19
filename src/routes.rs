use std::sync::Arc;
use axum::Json;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use axum::extract::State;
use axum::http::StatusCode;
use crate::embedding::{Embedder, SentenceTransformer};

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

// TODO: Remove unwraps
pub async fn text_embeddings(State(embedder): State<Arc<Embedder>>, Json(payload): Json<EmbeddingsRequest>)
                             -> Result<impl IntoResponse, TextEmbeddingRouteError> {
	let sentences: Vec<String> = payload.input.into();

	let sentences = sentences.iter().map(|s| s.as_str()).collect();
	let embeddings = embedder.encode_batch(sentences, true)?;

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
		model: payload.model.to_string(),
		// TODO: Calculate tokens
		usage: Usage {
			prompt_tokens: 0,
			total_tokens: 0,
		},
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

#[derive(Debug, Serialize, PartialEq, Default)]
pub struct Usage {
	pub prompt_tokens: u32,
	pub total_tokens: u32,
}
