use anyhow::Result;
use tokio::signal;

type Nullary = fn() -> Result<()>;

pub async fn shutdown_signal(shutdown_fns_opt: Option<&[Nullary]>) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };
    
    if let Some(shutdown_fns) = shutdown_fns_opt {
        for shutdown_fn in shutdown_fns {
            shutdown_fn().expect("Failed to call shutdown function.");
        }
    }

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
