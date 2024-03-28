pub mod queue;
mod client;
pub mod embed;
mod handler;

pub use queue::Queue;
use uuid::Uuid;

// Generic types for task-specific data
type TaskId = Uuid;

