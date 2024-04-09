use anyhow::Result;
use std::ops::RangeInclusive;
use std::result;
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

const PORT_RANGE: RangeInclusive<u16> = 1..=65535;

pub fn port_in_range(s: &str) -> result::Result<u16, String> {
    let port: u16 = s
        .parse()
        .map_err(|_| format!("`{s}` isn't a port number"))?;
    if PORT_RANGE.contains(&port) {
        Ok(port)
    } else {
        Err(format!(
            "port not in range {}-{}",
            PORT_RANGE.start(),
            PORT_RANGE.end()
        ))
    }
}
