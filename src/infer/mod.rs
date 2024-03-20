pub mod queue;
mod client;
pub mod embed;

pub use queue::Queue;
use uuid::Uuid;

// Marker trait for task requests
pub trait TaskRequest: Send + Sync + 'static {}

// Marker trait for task responses
pub trait TaskResponse: Send + Sync + 'static {}

// Generic types for task-specific data
type TaskId = Uuid;

