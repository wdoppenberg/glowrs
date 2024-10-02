use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Invalid model name: {0}")]
    InvalidModelName(&'static str),

    #[error("Model load error: {0}")]
    ModelLoad(&'static str),

    #[error("Invalid argument: {0}")]
    InvalidArgument(&'static str),

    #[error("Invalid model architecture: {0}")]
    InvalidModelConfig(&'static str),

    #[error("Inference error: {0}")]
    InferenceError(&'static str),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Tokenization error: {0}")]
    Tokenization(#[from] tokenizers::Error),

    #[error("Serde JSON error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),

    #[error("HF Hub error: {0}")]
    HFHub(#[from] hf_hub::api::sync::ApiError),

    #[error("Generic error: {0}")]
    Generic(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = Error::InvalidModelName("test");
        assert_eq!(error.to_string(), "Invalid model name: test");

        let error = Error::ModelLoad("test");
        assert_eq!(error.to_string(), "Model load error: test");

        let error = Error::InvalidModelConfig("test");
        assert_eq!(error.to_string(), "Invalid model architecture: test");

        let error = Error::Candle(candle_core::Error::UnexpectedNumberOfDims {
            shape: (32, 32).into(),
            expected: 3,
            got: 2,
        });
        assert_eq!(
            error.to_string(),
            "Candle error: unexpected rank, expected: 3, got: 2 ([32, 32])"
        );

        let error = Error::IO(std::io::Error::new(std::io::ErrorKind::Other, "test"));
        assert_eq!(error.to_string(), "IO error: test");

        let error = Error::HFHub(hf_hub::api::sync::ApiError::MissingHeader("test"));
        assert_eq!(error.to_string(), "HF Hub error: Header test is missing");
    }
}
