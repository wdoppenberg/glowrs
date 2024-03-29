use anyhow::Result;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use tokio::time::Instant;
use uuid::Uuid;
use crate::infer::handler::RequestHandler;

use super::TaskId;

/// Queue entry
#[derive(Debug)]
pub(crate) struct QueueEntry<TReq, TResp>
where
    TReq: Send + Sync + 'static,
    TResp: Send + Sync + 'static,
{
    /// Identifier
    pub id: TaskId,

    /// Request
    pub request: TReq,

    /// Response sender
    pub response_tx: oneshot::Sender<TResp>,

    /// Instant when this entry was queued
    pub queue_time: Instant,
}


/// Queue entry
impl<TReq, TResp> QueueEntry<TReq, TResp> where TReq: Send + Sync + 'static, TResp: Send + Sync + 'static {
    pub fn new(request: TReq, response_tx: oneshot::Sender<TResp>) -> Self {
        Self {
            id: Uuid::new_v4(),
            request,
            response_tx,
            queue_time: Instant::now(),
        }
    }
}

/// Queue command
#[derive(Debug)]
// TODO: Use stop command
#[allow(dead_code)]
pub(crate) enum QueueCommand<TReq, TResp>
where
    TReq: Send, TReq: Send + Sync + 'static, TResp: Send + Sync + 'static
{
    Append(QueueEntry<TReq, TResp>),
    Stop,
}

#[derive(Clone)]
struct CloneableJoinHandle<T: Clone>(Arc<Mutex<JoinHandle<T>>>);

impl<T> CloneableJoinHandle<T>
where T: Clone
{
    #[allow(dead_code)]
    fn new(handle: JoinHandle<T>) -> Self {
        Self(Arc::new(Mutex::new(handle)))
    }
}

/// Request Queue with stateful task processor
#[derive(Clone)]
pub struct Queue<THandler>
where
    THandler: RequestHandler,
{
    pub(crate) tx: UnboundedSender<QueueCommand<THandler::TReq, THandler::TResp>>,
    // _handle: CloneableJoinHandle<Result<()>>,
    _processor: PhantomData<THandler>,
}

impl<THandler> Queue<THandler>
where
    THandler: RequestHandler
{
    pub(crate) fn new(processor: THandler) -> Result<Self> {

        // Create channel
        let (queue_tx, queue_rx) = unbounded_channel();

        let _join_handle = std::thread::spawn(move || {

            // Create a new Runtime to run tasks
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name(format!("queue-{}", Uuid::new_v4()))
                .build()?;

            // Pull task requests off the channel and send them to the executor
            runtime.block_on(queue_task(queue_rx, processor))
        });

        Ok(Self {
            tx: queue_tx,
            // _handle: CloneableJoinHandle::new(_join_handle),
            _processor: PhantomData,
        })
    }
}

// Generic background task executor with stateful processor
async fn queue_task<THandler>(
    mut receiver: UnboundedReceiver<QueueCommand<THandler::TReq, THandler::TResp>>,
    mut processor: THandler,
) -> Result<()>
where
    THandler: RequestHandler
{
    'main: while let Some(cmd) = receiver.recv().await {
        use QueueCommand::*;

        match cmd {
            Append(entry) => {
                tracing::trace!(
                    "Processing task {}, added {}ms ago",
                    entry.id,
                    entry.queue_time.elapsed().as_millis()
                );

                // Process the task
                let response = processor.handle(entry.request)?;

                if entry.response_tx.send(response).is_ok() {
                    tracing::trace!("Successfully sent response for task {}", entry.id)
                } else {
                    tracing::error!("Failed to send response for task {}", entry.id)
                }
            }
            Stop => {
                tracing::info!("Stopping queue task");
                break 'main;
            }
        }
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[derive(Debug, PartialEq)]
    struct Task {
        name: String,
    }
    
    impl Task {
        fn new(name: String) -> Self {
            Self { name }
        }
    }
    
    struct TaskProcessor;
    
    impl TaskProcessor {
         fn new() -> Result<Self> {
            Ok(Self)
        }
    }
    
    impl RequestHandler for TaskProcessor {
        type TReq = Task;
        type TResp = Task;


        fn handle(&mut self, request: Task) -> Result<Task> {
            let new_name = format!("{}-processed", request.name);
            Ok(Task::new(new_name))
        }
    }

    #[tokio::test]
    async fn test_queue() {
        // Create a new processor
        let processor = TaskProcessor::new().unwrap();
        
        // Create a new queue
        let queue: Queue<TaskProcessor> = Queue::new(processor).unwrap();

        // Set a task name
        let name = "test".to_string();
        
        // Create a new task
        let task = Task::new(name.clone());

        // Send the task to the queue
        let (task_tx, task_rx) = oneshot::channel();
        queue.tx.send(QueueCommand::Append(QueueEntry::new(task, task_tx))).unwrap();

        // Wait for the response
        let response = task_rx.await.unwrap();
        assert_eq!(response, Task::new(format!("{}-processed", name).to_string()));
    }
}
