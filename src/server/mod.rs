mod init;
mod state;
pub mod routes;
pub mod utils;
pub mod data_models;

pub use init::init_router;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ServerError {
	#[error("Internal server error: `{0}`")]
	InternalServerError(#[from] anyhow::Error),
	
	#[error("Too many requests.")]
	TooManyRequestsError,
	
	#[error("Inference error")]
	InferenceError,
}

impl IntoResponse for ServerError {
	fn into_response(self) -> Response {
		match self {
    		ServerError::InternalServerError(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    		ServerError::TooManyRequestsError => StatusCode::TOO_MANY_REQUESTS.into_response(),
    		ServerError::InferenceError => StatusCode::BAD_REQUEST.into_response(),
    	}
	}
}

