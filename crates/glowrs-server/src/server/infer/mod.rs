pub mod batch;
mod client;
pub mod embed;
pub mod executor;
mod handler;

pub use executor::DedicatedExecutor;
use uuid::Uuid;

// Generic types for task-specific data
type TaskId = Uuid;
