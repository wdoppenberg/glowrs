use std::net::IpAddr;
use anyhow::Result;
use std::process::ExitCode;
use clap::Parser;
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use glowrs::model::device::print_device_info;

mod server;
use server::utils;
use server::{init_router, RouterArgs};
use server::utils::port_in_range;


#[derive(Debug, Parser)]
pub struct App {
    #[clap(flatten)]
    pub router_args: RouterArgs,

    #[arg(value_parser = port_in_range)]
    #[clap(short, long, default_value = "3000")]
    pub port: u16,
    
    #[clap(short, long, default_value = "127.0.0.1")]
    pub host: IpAddr,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<ExitCode> {
    let args = App::parse();

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                eprintln!("No environment variables found that can initialize tracing_subscriber::EnvFilter. Using defaults.");
                
                // axum logs rejections from built-in extractors with the `axum::rejection`
                // target, at `TRACE` level. `axum::rejection=trace` enables showing those events
                "glowrs=trace,server=debug,tower_http=debug,axum::rejection=trace".into()
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // TODO: Configuration passing
    print_device_info();

    let router = init_router(&args.router_args)?;

    let listener = TcpListener::bind(format!("{}:{}", args.host, args.port)).await?;
    tracing::info!("listening on {}", listener.local_addr()?);
    axum::serve(listener, router)
        .with_graceful_shutdown(utils::shutdown_signal(None))
        .await?;

    Ok(ExitCode::SUCCESS)
}