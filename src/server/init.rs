use axum::Router;
use std::sync::Arc;
use axum::routing::{get, post};
use tower_http::trace::TraceLayer;
use axum::http::Request;
use axum::extract::MatchedPath;
use tracing::{info_span, Span};
use tower_http::timeout::TimeoutLayer;
use std::time::Duration;
use clap::Args;
use thiserror::__private::AsDisplay;

use crate::infer::embed::EmbeddingsHandler;
use crate::server::routes::{default, embeddings};
use crate::server::state::ServerState;

#[derive(Debug, Args)]
pub struct RouterArgs {
    #[clap(short, long)]
    pub model_repo: String,
    
    #[clap(short, long, default_value = "main")]
    pub revision: String,
}

pub fn init_router(args: &RouterArgs) -> anyhow::Result<Router> {
    let embeddings_handler = EmbeddingsHandler::new(
        &args.model_repo,
        &args.revision,
    )?;
    
    let state = Arc::new(ServerState::new(embeddings_handler)?);

    let router = Router::new()
        .route("/v1/embeddings", post(embeddings::infer_text_embeddings))
        .route("/health", get(default::health_check))
        .with_state(state)
        .layer((
            TraceLayer::new_for_http()
                .make_span_with(|request: &Request<_>| {
                    // Log the matched route's path (with placeholders not filled in).
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
    Ok(router)
}
