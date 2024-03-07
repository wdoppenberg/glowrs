use axum::extract::State;
use std::sync::Arc;
use axum::Json;
use tokio::time::Instant;
use axum::http::StatusCode;
use crate::server::routes::{EmbeddingsRequest, EmbeddingsResponse, TextEmbeddingRouteError};
use crate::server::state::ServerState;


pub async fn text_embeddings(
    State(server_state): State<Arc<ServerState>>,
    Json(embeddings_request): Json<EmbeddingsRequest>,
) -> Result<(StatusCode, Json<EmbeddingsResponse>), TextEmbeddingRouteError>
{
    let start = Instant::now();
    let response = server_state.infer.generate_embedding(embeddings_request).await?;

    let duration = Instant::now() - start;
    tracing::trace!("Inference took {} ms", duration.as_millis());
    
    Ok((StatusCode::OK, Json(response)))
}
