use axum::routing::post;
use axum::{http::StatusCode, routing::get, Router};
use glowrs::server;
use glowrs::work::queue::Queue;
use glowrs::work::task::TaskID;
use glowrs::work::Task;
use std::net::SocketAddr;
use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;

async fn health_check() -> StatusCode {
    tracing::info!("Health check request received.");
    StatusCode::OK
}

#[derive(Debug, Clone)]
struct ExampleTask {
    input: String,
}

impl Task for ExampleTask {
    type Input = String;
    type Output = ();

    fn process(&self) -> Self::Output {
        println!("Processing {}", self.input);
        sleep(Duration::from_secs(2));
    }

    fn get_id(&self) -> TaskID {
        TaskID::new_v4()
    }
}
#[tokio::main]
async fn main() {
    // initialize tracing
    // let subscriber = tracing_subscriber::FmtSubscriber::builder()
    //     .with_max_level(tracing::Level::TRACE) // Set the maximum tracing level
    //     .finish();
    //
    // tracing::subscriber::set_global_default(subscriber).expect("Failed to set global subscriber");

    console_subscriber::init();

    let q = Arc::new(Queue::<ExampleTask>::new());

    // build our application with a route
    let app = Router::new()
        // `GET /` goes to `root`
        .route("/create-task", post(server::create_task))
        // .route("/delete-task/:id", get(queue::delete_task))
        .route("/clear-tasks", get(server::clear_tasks))
        .route("/process-tasks", get(server::process_tasks))
        .route("/health", get(health_check))
        .with_state(q);

    // run our app with hyper
    // `axum::Server` is a re-export of `hyper::Server`
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::debug!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();

    axum::serve(listener, app).await.unwrap();
}
