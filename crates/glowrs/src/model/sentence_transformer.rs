use crate::config::model::ModelType;
use crate::model::embedder::{
    encode_batch, encode_batch_with_usage, load_pretrained_model, EmbedOutput, EmbedderModel,
};
use crate::model::utils;
use crate::{Device, Error, Result};
use candle_core::Tensor;
use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::{Repo, RepoType};
use std::path::Path;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::EncodeInput;

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
///  let embeddings = encoder.encode_batch(sentences, normalize).unwrap();
///
///  println!("{:?}", embeddings);
///  ```
///
pub struct SentenceTransformer {
    model: Box<dyn EmbedderModel>,
    tokenizer: Tokenizer,
    model_type: ModelType,
}

impl SentenceTransformer {
    pub fn new(model: Box<dyn EmbedderModel>, tokenizer: Tokenizer, model_type: ModelType) -> Self {
        Self {
            model,
            tokenizer,
            model_type,
        }
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
        let span = tracing::span!(tracing::Level::TRACE, "st-from-repo-string");
        let _enter = span.enter();
        let (model_repo, default_revision) = utils::parse_repo_string(repo_string)?;
        Self::from_repo(model_repo, default_revision, device)
    }

    pub fn from_repo(repo_name: &str, revision: &str, device: &Device) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "st-from-repo");
        let _enter = span.enter();
        let api = Api::new()?.repo(Repo::with_revision(
            repo_name.into(),
            RepoType::Model,
            revision.into(),
        ));

        Self::from_api(api, device)
    }

    pub fn from_api(api: ApiRepo, device: &Device) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "st-from-api");
        let _enter = span.enter();
        let model_path = api.get("model.safetensors")?;

        let _config_path = api.get("config.json")?;

        let _tokenizer_path = api.get("tokenizer.json")?;

        let _pooling_dir_opt = api.get("1_Pooling/config.json").ok();
        if _pooling_dir_opt.is_none() {
            tracing::info!("No pooling configuration found. Using default or given strategy.");
        }

        // TODO: Remove expect
        let model_root = model_path
            .parent()
            .expect("Model path has no parent directory");

        Self::from_path(model_root, device)
    }

    pub fn from_path(model_root: &Path, device: &Device) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "st-from-path");
        let _enter = span.enter();
        let tokenizer_path = model_root.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)?;

        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        let (model, model_type) = load_pretrained_model(model_root, device)?;

        Ok(Self::new(model, tokenizer, model_type))
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
        let span = tracing::span!(tracing::Level::TRACE, "st-from-folder");
        let _enter = span.enter();

        // Construct PathBuf objects for model, config, and tokenizer json files
        let model_path = folder_path.join("model.safetensors");
        let config_path = folder_path.join("config.json");
        let tokenizer_path = folder_path.join("tokenizer.json");

        if !model_path.exists() || !config_path.exists() || !tokenizer_path.exists() {
            Err(Error::ModelLoad(
                "model.safetensors, config.json, or tokenizer.json does not exist in the given directory"
            ))
        } else {
            Self::from_path(&model_path, device)
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
    /// let encoder = SentenceTransformer::from_repo_string("sentence-transformers/all-MiniLM-L6-v2", &Device::Cpu)?;
    ///
    /// # Ok(())
    /// # }
    ///
    pub fn encode_batch_with_usage<'s, E>(
        &self,
        sentences: Vec<E>,
        normalize: bool,
    ) -> Result<EmbedOutput>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let span = tracing::span!(tracing::Level::TRACE, "st-encode-batch");
        let _enter = span.enter();

        encode_batch_with_usage(
            self.model.as_ref(),
            &self.tokenizer,
            sentences,
            &self.model_type,
            normalize,
        )
    }

    pub fn encode_batch<'s, E>(&self, sentences: Vec<E>, normalize: bool) -> Result<Tensor>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let span = tracing::span!(tracing::Level::TRACE, "st-encode-batch");
        let _enter = span.enter();

        encode_batch(
            self.model.as_ref(),
            &self.tokenizer,
            sentences,
            &self.model_type,
            normalize,
        )
    }

    pub fn get_tokenizer_mut(&mut self) -> &mut Tokenizer {
        &mut self.tokenizer
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;
//     use std::time::Instant;
//
//     const BERT_TOKENIZER_PATH: &str = "tests/fixtures/all-MiniLM-L6-v2/tokenizer.json";
//     const BERT_CONFIG_PATH: &str = "tests/fixtures/all-MiniLM-L6-v2/config.json";
//
//     fn test_sentence_transformer(config_path: &str, tokenizer_path: &str) -> Result<()> {
//         let device = &Device::Cpu;
//         let sentence_transformer: SentenceTransformer = SentenceTransformer::test_from_config_json(
//             Path::new(config_path),
//             Path::new(tokenizer_path),
//             device,
//         )?;
//
//         let sentences = vec![
//             "The cat sits outside",
//             "A man is playing guitar",
//             "I love pasta",
//             "The new movie is awesome",
//             "The cat plays in the garden",
//             "A woman watches TV",
//             "The new movie is so great",
//             "Do you like pizza?",
//         ];
//
//         let pooling_strategy = PoolingStrategy::Mean;
//
//         let start = Instant::now();
//         let embeddings = sentence_transformer.encode_batch(sentences, true, pooling_strategy)?;
//
//         println!("Pooled embeddings {:?}", embeddings.shape());
//         println!(
//             "Inference done in {}ms",
//             (Instant::now() - start).as_millis()
//         );
//
//         Ok(())
//     }
//
//     #[test]
//     fn test_sentence_transformer_bert() -> Result<()> {
//         test_sentence_transformer(BERT_CONFIG_PATH, BERT_TOKENIZER_PATH)?;
//
//         Ok(())
//     }
// }
