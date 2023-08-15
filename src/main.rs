use axum::{http::StatusCode, Router, routing::get};
use std::net::SocketAddr;
use axum::routing::post;
use queue::Queue;

mod queue;
mod worker;


async fn health_check() -> StatusCode {
    tracing::info!("Health check request received.");
    StatusCode::OK
}

#[tokio::main]
async fn main() {
    // initialize tracing
    tracing_subscriber::fmt::init();

    let queue = Queue::new(4);

    // build our application with a route
    let app = Router::new()
        // `GET /` goes to `root`
        .route("/create-task", post(queue::create_task))
        // .route("/delete-task/:id", get(queue::delete_task))
        .route("/clear-tasks", get(queue::clear_tasks))
        .route("/process-tasks", get(queue::process_tasks))
        .route("/health", get(health_check))
        .with_state(queue);


    // run our app with hyper
    // `axum::Server` is a re-export of `hyper::Server`
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::debug!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

