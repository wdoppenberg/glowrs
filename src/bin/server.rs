use std::process::ExitCode;
use std::sync::Arc;
use axum::{http, Router};
use tokio::net::TcpListener;
use anyhow::Result;
use axum::extract::MatchedPath;
use axum::http::Request;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use thiserror::__private::AsDisplay;
use tower_http::trace::TraceLayer;
use tracing::{info_span, Span};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use glowrs::embedding::sbert::JinaBertBaseV2;
use glowrs::embedding::sentence_transformer::{Args, SentenceTransformer};

use glowrs::routes::text_embeddings;

async fn health_check() -> impl IntoResponse {
	(http::StatusCode::OK, "Everything is ok!".to_string())
}

#[tokio::main]
async fn main() -> Result<ExitCode> {
	tracing_subscriber::registry()
		.with(
			tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
				// axum logs rejections from built-in extractors with the `axum::rejection`
				// target, at `TRACE` level. `axum::rejection=trace` enables showing those events
				"server=debug,tower_http=debug,axum::rejection=trace".into()
			}),
		)
		.with(tracing_subscriber::fmt::layer())
		.init();

	tracing::info!("Loading embedder");
	let embedder: SentenceTransformer<JinaBertBaseV2> = SentenceTransformer::try_new()?;

	let state = Arc::new(embedder);

	let app = Router::new()
		.route("/v1/embeddings", post(text_embeddings))
		.route("/health",get(health_check))
        .with_state(state)
        .layer(
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
			})
	);

	let listener = TcpListener::bind("127.0.0.1:3000").await?;
	tracing::info!("listening on {}", listener.local_addr()?);
	axum::serve(listener, app).await?;

	Ok(ExitCode::SUCCESS)
}