pub mod queue;
pub mod task;
pub mod worker;
pub use task::Task;

#[cfg(test)]
mod tests {
    use crate::work::queue::Queue;
    use crate::work::task::{Task, TaskID};
    use std::thread::sleep;
    use std::time::Duration;

    #[derive(Debug, Clone)]
    pub struct ExampleTask {
        pub(crate) input: String,
    }

    impl Task for ExampleTask {
        type Input = String;
        type Output = ();

        fn process(&self) -> Self::Output {
            println!("Processing {}", self.input);
            sleep(Duration::from_millis(100));
        }

        fn get_id(&self) -> TaskID {
            TaskID::new_v4()
        }
    }

    #[tokio::test]
    async fn test_queue() {
        let queue = Queue::new();

        let tasks = (0..2)
            .map(|i| {
                let description = format!("Task {}", i);
                Box::new(ExampleTask { input: description })
            })
            .collect::<Vec<_>>();

        for task in tasks {
            queue.append(task).expect("Failed to append task");
        }

        queue.join().await.expect("Failed to join queue");
    }
}
