use axum::extract::State;
use std::sync::Arc;
use axum::Json;
use tokio::time::Instant;
use axum::http::StatusCode;
use anyhow::Result;

use crate::server::data_models::{EmbeddingsRequest, EmbeddingsResponse};
use crate::server::ServerError;
use crate::server::state::ServerState;


pub async fn infer_text_embeddings(
    State(server_state): State<Arc<ServerState>>,
    Json(embeddings_request): Json<EmbeddingsRequest>,
) -> Result<(StatusCode, Json<EmbeddingsResponse>), ServerError>
{
    let start = Instant::now();
    let response = server_state.embeddings_client.generate_embedding(embeddings_request).await?;

    let duration = Instant::now() - start;
    tracing::trace!("Inference took {} ms", duration.as_millis());
    
    Ok((StatusCode::OK, Json(response)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::Instant;
    use std::sync::Arc;
    use anyhow::Context;
    use crate::infer::Queue;
    use crate::server::data_models::{EncodingFormat::Float, Sentences};

    #[tokio::test]
    async fn test_text_embeddings_request() -> Result<()> {
        let queue = Queue::new()?;
        let server_state = Arc::new(
            ServerState::new(&queue)
                .context("Failed to create server state")?
        );
        let embeddings_request = EmbeddingsRequest {
            input: Sentences::from(Vec::from(["sentence sentence sentence"; 5])),
            model: "whatever".to_string(),
            encoding_format: Float
        }; 

        let start = Instant::now();
        const N_ITERS: usize = 2; 
        for _ in 0..N_ITERS { // number of iterations, adjust as required
            let _ = infer_text_embeddings(State(server_state.clone()), Json(embeddings_request.clone())).await;
        }
        let duration = Instant::now() - start;

        println!("Processing {} iterations took {} ms", N_ITERS, duration.as_millis());

        Ok(())
    }
}