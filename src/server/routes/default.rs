use axum::{http, response::IntoResponse};

pub async fn health_check() -> impl IntoResponse {
    (http::StatusCode::OK, "Everything is ok!".to_string())
}
