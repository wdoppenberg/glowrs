pub mod queue;
pub mod worker;
pub mod task;
pub use task::Task;

#[cfg(test)]
mod tests {
    use std::thread::sleep;
    use std::time::Duration;
    use crate::proc::queue::Queue;
    use crate::proc::task::{Task, TaskID};


    #[derive(Debug, Clone)]
    pub struct ExampleTask {
        input: String,
    }

    impl Task for ExampleTask {
        type Input = String;
        type Output = ();

        fn from_input(input: Self::Input) -> Self {
            Self { input }
        }

        fn process(&self) -> Self::Output {
            println!("Processing {}", self.input);
            sleep(Duration::from_secs(2));
        }

        fn get_id(&self) -> TaskID {
            TaskID::new_v4()
        }
    }

    #[tokio::test]
    async fn test_queue() {
        let queue = Queue::new(2);

        let tasks = (0..2).map(|i| {
            let description = format!("Task {}", i);
            Box::new(ExampleTask::from_input(description))
        }).collect::<Vec<_>>();

        for task in tasks {
            queue.append(task).expect("Failed to append task");
        }

        queue.join().await.expect("Failed to join queue");
    }
}
