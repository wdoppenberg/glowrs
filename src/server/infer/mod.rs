pub mod executor;
mod client;
pub mod embed;
mod handler;
pub mod batch;

use uuid::Uuid;
pub use executor::DedicatedExecutor;
pub use handler::RequestHandler;

// Generic types for task-specific data
type TaskId = Uuid;

