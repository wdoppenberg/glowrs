use crate::model::device::DEVICE;
use anyhow::{Context, Error, Result};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::BertModel;
use candle_transformers::models::jina_bert::BertModel as JinaBertModel;
use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::{Repo, RepoType};
use std::path::Path;
use tokenizers::tokenizer::Tokenizer;

use crate::model::embedder::{
    encode_batch, encode_batch_with_usage, EmbedderModel, EmbedderType, LoadableModel,
};
use crate::model::utils;
use crate::Sentences;
use crate::Usage;

/// The SentenceTransformer struct is the main entry point for using pre-trained models for embeddings and sentence similarity.
///
/// ## Example
///
/// ```rust
///  # use glowrs::SentenceTransformer;
///
///  let encoder = SentenceTransformer::from_repo_string("sentence-transformers/all-MiniLM-L6-v2").unwrap();
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
}

impl SentenceTransformer {
    pub fn new(model: Box<dyn EmbedderModel>, tokenizer: Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// Load a SentenceTransformer model from the Hugging Face Hub.
    ///
    /// ## Example
    ///
    /// ```rust
    /// # use glowrs::SentenceTransformer;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let encoder = SentenceTransformer::from_repo_string("sentence-transformers/all-MiniLM-L6-v2")?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// If you want to use a specific revision of the model, you can specify it in the repo string:
    ///
    ///
    pub fn from_repo_string(repo_string: &str) -> Result<Self> {
        let (model_repo, default_revision, embedder_type) = utils::parse_repo_string(repo_string)?;
        Self::from_repo(model_repo, default_revision, embedder_type)
    }

    pub fn from_repo(repo_name: &str, revision: &str, embedder_type: EmbedderType) -> Result<Self> {
        let api = Api::new()?.repo(Repo::with_revision(
            repo_name.into(),
            RepoType::Model,
            revision.into(),
        ));

        Self::from_api(api, embedder_type)
    }

    fn from_api_gen<L>(api: ApiRepo) -> Result<Self>
    where
        L: LoadableModel,
    {
        let model_path = api
            .get("model.safetensors")
            .context("Model repository is not available or doesn't contain `model.safetensors`.")?;

        let config_path = api
            .get("config.json")
            .context("Model repository doesn't contain `config.json`.")?;

        let tokenizer_path = api
            .get("tokenizer.json")
            .context("Model repository doesn't contain `tokenizer.json`.")?;

        Self::from_path_gen::<L>(&model_path, &config_path, &tokenizer_path)
    }

    fn from_path_gen<L>(
        model_path: &Path,
        config_path: &Path,
        tokenizer_path: &Path,
    ) -> Result<Self>
    where
        L: LoadableModel,
    {
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(Error::msg)?;
        let config_str = std::fs::read_to_string(config_path)?;

        let cfg = serde_json::from_str(&config_str)
            .context(
                "Failed to deserialize config.json. Make sure you have the right EmbedderModel implementation."
            )?;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &DEVICE)? };

        let model =
            L::load_model(vb, &cfg).context("Something went wrong while loading the model.")?;

        Ok(Self::new(model, tokenizer))
    }

    pub fn from_path(
        model_path: &Path,
        config_path: &Path,
        tokenizer_path: &Path,
        embedder_type: EmbedderType,
    ) -> Result<Self> {
        match embedder_type {
            EmbedderType::Bert => {
                Self::from_path_gen::<BertModel>(model_path, config_path, tokenizer_path)
            }
            EmbedderType::JinaBert => {
                Self::from_path_gen::<JinaBertModel>(model_path, config_path, tokenizer_path)
            }
        }
    }

    pub fn from_folder(folder_path: &Path, embedder_type: EmbedderType) -> Result<Self> {
        // Construct PathBuf objects for model, config, and tokenizer json files
        let model_path = folder_path.join("model.safetensors");
        let config_path = folder_path.join("config.json");
        let tokenizer_path = folder_path.join("tokenizer.json");

        if !model_path.exists() || !config_path.exists() || !tokenizer_path.exists() {
            Err(anyhow::anyhow!("model.safetensors, config.json, or tokenizer.json does not exist in the given directory"))
        } else {
            Self::from_path(&model_path, &config_path, &tokenizer_path, embedder_type)
        }
    }

    pub fn from_api(api: ApiRepo, embedder_type: EmbedderType) -> Result<Self> {
        match embedder_type {
            EmbedderType::Bert => Self::from_api_gen::<BertModel>(api),
            EmbedderType::JinaBert => Self::from_api_gen::<JinaBertModel>(api),
        }
    }

    pub fn encode_batch_with_usage(
        &self,
        sentences: impl Into<Sentences>,
        normalize: bool,
    ) -> Result<(Tensor, Usage)> {
        let (embeddings, usage) = encode_batch_with_usage(
            self.model.as_ref(),
            &self.tokenizer,
            sentences.into(),
            normalize,
        )?;
        Ok((embeddings, usage))
    }

    pub fn encode_batch(&self, sentences: impl Into<Sentences>, normalize: bool) -> Result<Tensor> {
        encode_batch(
            self.model.as_ref(),
            &self.tokenizer,
            sentences.into(),
            normalize,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_sentence_transformer() -> Result<()> {
        let start = Instant::now();

        let model_repo = "sentence-transformers/all-MiniLM-L6-v2";
        let default_revision = "main";
        let sentence_transformer: SentenceTransformer =
            SentenceTransformer::from_repo(model_repo, default_revision, EmbedderType::Bert)?;

        let sentences = Sentences::from(vec![
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
        ]);

        let model_load_duration = Instant::now() - start;
        println!("Model loaded in {}ms", model_load_duration.as_millis());

        let embeddings = sentence_transformer.encode_batch(sentences, true)?;

        println!("Pooled embeddings {:?}", embeddings.shape());
        println!(
            "Inference done in {}ms",
            (Instant::now() - start - model_load_duration).as_millis()
        );

        Ok(())
    }
    
    #[test]
    fn test_from_folder() -> Result<()> {
        // TODO: Expand user (tilde)
        let path = Path::new("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/0b6dc4ef7c29dba0d2e99a5db0c855c3102310d8/");
        
        let _encoder = SentenceTransformer::from_folder(&path, EmbedderType::Bert)?;
        
        Ok(())
    }
}
