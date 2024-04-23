use candle_core::Tensor;
use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::{Repo, RepoType};
use std::path::Path;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::EncodeInput;

use crate::model::embedder::{
    encode_batch, encode_batch_with_usage, load_pretrained_model, EmbedderModel,
};
use crate::model::utils;
use crate::{Device, Error, Result, Usage};

#[cfg(test)]
use crate::model::embedder::{load_zeros_model, parse_config};
use crate::model::pooling::PoolingStrategy;

/// The SentenceTransformer struct is the main entry point for using pre-trained models for embeddings and sentence similarity.
///
/// ## Example
///
/// ```rust
/// # use candle_core::Device;
/// use glowrs::{SentenceTransformer, PoolingStrategy};
///
///  let encoder = SentenceTransformer::from_repo_string("sentence-transformers/all-MiniLM-L6-v2", &Device::Cpu).unwrap();
///
///  let sentences = vec![
///     "Hello, how are you?",
///     "Hey, how are you doing?"
///  ];
///
///  let normalize = false;
///  let embeddings = encoder.encode_batch(sentences, normalize, PoolingStrategy::Mean).unwrap();
///
///  println!("{:?}", embeddings);
///  ```
///
pub struct SentenceTransformer {
    model: Box<dyn EmbedderModel>,
    tokenizer: Tokenizer,
}

impl SentenceTransformer {
    pub fn new(model: Box<dyn EmbedderModel>, tokenizer: Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// Load a [`SentenceTransformer`] model from the Hugging Face Hub.
    ///
    /// ## Example
    ///
    /// ```rust
    /// # use glowrs::{SentenceTransformer, Device, Error};
    ///
    /// # fn main() -> Result<(), Error> {
    /// use glowrs::Error;
    /// let encoder = SentenceTransformer::from_repo_string("sentence-transformers/all-MiniLM-L6-v2", &Device::Cpu)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_repo_string(repo_string: &str, device: &Device) -> Result<Self> {
        let (model_repo, default_revision) = utils::parse_repo_string(repo_string)?;
        Self::from_repo(model_repo, default_revision, device)
    }

    pub fn from_repo(repo_name: &str, revision: &str, device: &Device) -> Result<Self> {
        let api = Api::new()?.repo(Repo::with_revision(
            repo_name.into(),
            RepoType::Model,
            revision.into(),
        ));

        Self::from_api(api, device)
    }

    pub fn from_api(api: ApiRepo, device: &Device) -> Result<Self> {
        let model_path = api.get("model.safetensors")?;

        let config_path = api.get("config.json")?;

        let tokenizer_path = api.get("tokenizer.json")?;

        Self::from_path(&model_path, &config_path, &tokenizer_path, device)
    }

    pub fn from_path(
        model_path: &Path,
        config_path: &Path,
        tokenizer_path: &Path,
        device: &Device,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        let model = load_pretrained_model(model_path, config_path, device)?;

        Ok(Self::new(model, tokenizer))
    }

    /// Load a [`SentenceTransformer`] model from a folder containing the model, config, and tokenizer
    /// json files. The model should be saved in the SafeTensors format. Often, these folders
    /// are created by huggingface libraries when pulling a model from the hub, and are saved in
    /// the `~/.cache/huggingface/hub/models` directory.
    ///
    /// ## Example
    ///
    /// ```no_run
    /// # use glowrs::{SentenceTransformer, Device};
    /// # use std::path::Path;
    ///
    /// # type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    ///
    /// # fn main() -> Result<()> {
    /// let path = Path::new("path/to/folder");
    ///
    /// let encoder = SentenceTransformer::from_folder(path, &Device::Cpu)?;
    ///
    /// # Ok(())
    /// # }
    pub fn from_folder(folder_path: &Path, device: &Device) -> Result<Self> {
        // Construct PathBuf objects for model, config, and tokenizer json files
        let model_path = folder_path.join("model.safetensors");
        let config_path = folder_path.join("config.json");
        let tokenizer_path = folder_path.join("tokenizer.json");

        if !model_path.exists() || !config_path.exists() || !tokenizer_path.exists() {
            Err(Error::ModelLoad(
                "model.safetensors, config.json, or tokenizer.json does not exist in the given directory"
            ))
        } else {
            Self::from_path(&model_path, &config_path, &tokenizer_path, device)
        }
    }

    /// Set the pooling strategy to use when encoding sentences.
    ///
    /// ## Example
    ///
    /// ```no_run
    /// # use glowrs::{SentenceTransformer, Device};
    /// # use glowrs::PoolingStrategy;
    ///
    /// # type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    ///
    /// # fn main() -> Result<()> {
    /// let encoder = SentenceTransformer::from_repo_string("sentence-transformers/all-MiniLM-L6-v2", &Device::Cpu)?
    ///    .with_pooling_strategy(PoolingStrategy::Sum);
    ///
    /// # Ok(())
    /// # }
    ///

    #[cfg(test)]
    pub(crate) fn test_from_config_json(
        config_path: &Path,
        tokenizer_path: &Path,
        device: &Device,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        let config_str = std::fs::read_to_string(config_path)?;

        let model_config = parse_config(&config_str)?;

        let model = load_zeros_model(model_config, device)?;

        Ok(Self::new(model, tokenizer))
    }

    pub fn encode_batch_with_usage<'s, E>(
        &self,
        sentences: Vec<E>,
        normalize: bool,
        pooling_strategy: PoolingStrategy,
    ) -> Result<(Tensor, Usage)>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let (embeddings, usage) = encode_batch_with_usage(
            self.model.as_ref(),
            &self.tokenizer,
            sentences,
            pooling_strategy,
            normalize,
        )?;
        Ok((embeddings, usage))
    }

    pub fn encode_batch<'s, E>(
        &self,
        sentences: Vec<E>,
        normalize: bool,
        pooling_strategy: PoolingStrategy,
    ) -> Result<Tensor>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        encode_batch(
            self.model.as_ref(),
            &self.tokenizer,
            sentences,
            pooling_strategy,
            normalize,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::time::Instant;

    const BERT_TOKENIZER_PATH: &str = "tests/fixtures/all-MiniLM-L6-v2/tokenizer.json";
    const BERT_CONFIG_PATH: &str = "tests/fixtures/all-MiniLM-L6-v2/config.json";

    fn test_sentence_transformer(config_path: &str, tokenizer_path: &str) -> Result<()> {
        let device = &Device::Cpu;
        let sentence_transformer: SentenceTransformer = SentenceTransformer::test_from_config_json(
            Path::new(config_path),
            Path::new(tokenizer_path),
            device,
        )?;

        let sentences = vec![
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
        ];

        let pooling_strategy = PoolingStrategy::Mean;

        let start = Instant::now();
        let embeddings = sentence_transformer.encode_batch(sentences, true, pooling_strategy)?;

        println!("Pooled embeddings {:?}", embeddings.shape());
        println!(
            "Inference done in {}ms",
            (Instant::now() - start).as_millis()
        );

        Ok(())
    }

    #[test]
    fn test_sentence_transformer_bert() -> Result<()> {
        test_sentence_transformer(BERT_CONFIG_PATH, BERT_TOKENIZER_PATH)?;

        Ok(())
    }
}
