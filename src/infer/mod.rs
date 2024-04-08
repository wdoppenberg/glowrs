pub mod queue;
mod client;
pub mod embed;
mod handler;
pub mod batch;

use uuid::Uuid;
pub use queue::Queue;

// Generic types for task-specific data
type TaskId = Uuid;

