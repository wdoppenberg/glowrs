use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;

use crate::server::ServerError;
use crate::server::state::ServerState;


#[derive(Debug, Serialize)]
pub struct ModelCard {
	id: String,
	object: String,
	created: usize,
	owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct ModelCardList {
	object: String,
	data: Vec<ModelCard>,
}

pub async fn list_models(
    State(server_state): State<Arc<ServerState>>,
) -> anyhow::Result<(StatusCode, Json<ModelCardList>), ServerError> {
	let model_map = &server_state.model_map;
	
	let model_cards = model_map
		.keys()
		.map(|model_name| {
			ModelCard {
				id: model_name.clone(),
				object: "model".to_string(),
				// This is a placeholder for the actual creation time
				created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as usize,
				owned_by: "hf_hub".to_string(),
			}
		}).collect();
	
	let model_card_list = ModelCardList {
		object: "list".to_string(),
		data: model_cards,
	};
	
	Ok((StatusCode::OK, Json(model_card_list)))
}

pub async fn get_model(
	State(server_state): State<Arc<ServerState>>,
	model_id: String,
) -> anyhow::Result<(StatusCode, Json<ModelCard>), ServerError> {
	let _ = server_state.model_map.get(&model_id)
		.ok_or(ServerError::ModelNotFound)?;
	
	let model_card = ModelCard {
		id: model_id.clone(),
		object: "model".to_string(),
		// This is a placeholder for the actual creation time
		created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as usize,
		owned_by: "hf_hub".to_string(),
	};
	
	Ok((StatusCode::OK, Json(model_card)))
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::sync::Arc;
	use anyhow::Context;

	#[tokio::test]
	async fn test_list_models() -> anyhow::Result<()> {
		let server_state = Arc::new(
			ServerState::new(vec!["jinaai/jina-embeddings-v2-base-en".to_string()])
				.context("Failed to create server state")?
		);
		let (status, model_card_list) = list_models(State(server_state)).await?;
		assert_eq!(status, StatusCode::OK);
		assert_eq!(model_card_list.data.len(), 1);
		assert_eq!(model_card_list.data[0].id, "jinaai/jina-embeddings-v2-base-en");
		Ok(())
	}
	
	#[tokio::test]
	async fn test_get_model() -> anyhow::Result<()> {
		let server_state = Arc::new(
			ServerState::new(vec!["jinaai/jina-embeddings-v2-base-en".to_string()])
				.context("Failed to create server state")?
		);
		let (status, model_card) = get_model(State(server_state), "jinaai/jina-embeddings-v2-base-en".to_string()).await?;
		assert_eq!(status, StatusCode::OK);
		assert_eq!(model_card.id, "jinaai/jina-embeddings-v2-base-en");
		Ok(())
	}
}