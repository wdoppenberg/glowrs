use anyhow::Result;
use std::process::ExitCode;
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use glowrs::utils::device::print_device_info;
use glowrs::server::{init_router, utils};

#[tokio::main(flavor = "multi_thread", worker_threads = 10)]
async fn main() -> Result<ExitCode> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                // axum logs rejections from built-in extractors with the `axum::rejection`
                // target, at `TRACE` level. `axum::rejection=trace` enables showing those events
                "glowrs=trace,server=debug,tower_http=debug,axum::rejection=trace".into()
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // TODO: Configuration passing
    print_device_info();

    let router = init_router();

    let listener = TcpListener::bind("127.0.0.1:3000").await?;
    tracing::info!("listening on {}", listener.local_addr()?);
    axum::serve(listener, router)
        .with_graceful_shutdown(utils::shutdown_signal())
        .await?;

    Ok(ExitCode::SUCCESS)
}