use axum::Router;
use std::sync::Arc;
use axum::routing::{get, post};
use tower_http::trace::TraceLayer;
use axum::http::Request;
use axum::extract::MatchedPath;
use tracing::{info_span, Span};
use tower_http::timeout::TimeoutLayer;
use thiserror::__private::AsDisplay;
use std::time::Duration;

use crate::server::routes::{default, text_embeddings};
use crate::server::state::ServerState;

pub fn init_router() -> Router {
    let state = Arc::new(ServerState::default());

    let router = Router::new()
        .route("/v1/embeddings", post(text_embeddings::text_embeddings))
        .route("/health", get(default::health_check))
        .with_state(state)
        .layer((
            TraceLayer::new_for_http()
                .make_span_with(|request: &Request<_>| {
                    // Log the matched route's path (with placeholders not filled in).
                    // Use request.uri() or OriginalUri if you want the real path.
                    let matched_path = request
                        .extensions()
                        .get::<MatchedPath>()
                        .map(MatchedPath::as_str);
                    tracing::debug!("{}", request.uri().as_display());

                    info_span!(
                        "http_request",
                        method = ?request.method(),
                        matched_path,
                        some_other_field = tracing::field::Empty,
                    )
                })
                .on_request(|_request: &Request<_>, _span: &Span| {
                    // You can use `_span.record("some_other_field", value)` in one of these
                    // closures to attach a value to the initially empty field in the info_span
                    // created above.
                }),
            TimeoutLayer::new(Duration::from_secs(15))
        ));
    router
}
