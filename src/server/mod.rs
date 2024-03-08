pub mod routes;
mod state;
pub mod utils;
mod router;
pub mod data_models;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror::Error;
pub use router::init_router;


#[derive(Error, Debug)]
pub enum ServerError {
	#[error("Internal server error: `{0}`")]
	InternalServerError(String),
	
	#[error("Too many requests.")]
	TooManyRequestsError,
	
	#[error("Inference error")]
	InferenceError,
	
	#[error("Unknown error")]
	Unknown(#[from] anyhow::Error)
}

impl IntoResponse for ServerError {
	fn into_response(self) -> Response {
		match self {
    		ServerError::InternalServerError(err) => (StatusCode::INTERNAL_SERVER_ERROR, err).into_response(),
    		ServerError::TooManyRequestsError => StatusCode::TOO_MANY_REQUESTS.into_response(),
    		ServerError::InferenceError => StatusCode::BAD_REQUEST.into_response(),
			ServerError::Unknown(err) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{}", err)).into_response()
    	}
	}
}

