pub mod executor;
mod client;
pub mod embed;
mod handler;
pub mod batch;

use uuid::Uuid;
pub use executor::DedicatedExecutor;

// Generic types for task-specific data
type TaskId = Uuid;

