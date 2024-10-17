use crate::core::config::model::{ModelType, SentenceTransformerConfig};
use crate::core::embedder::{
    encode_batch, encode_batch_with_usage, load_pretrained_model, EmbedOutput, EmbedderModel,
};
use crate::core::repo::{ModelRepo, ModelRepoFiles};
use crate::{Device, Error, PoolingStrategy, Result};

use crate::core::utils;
use candle_core::Tensor;
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use std::marker::PhantomData;
use std::path::Path;
use std::str::FromStr;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::{EncodeInput, Encoding};

/// The SentenceTransformer struct is the main abstraction for using pre-trained models for
/// generating text embeddings.
pub struct SentenceTransformer {
    model: Box<dyn EmbedderModel>,
    tokenizer: Tokenizer,
    model_type: ModelType,
}

impl SentenceTransformer {
    pub(crate) fn new(
        model: Box<dyn EmbedderModel>,
        tokenizer: Tokenizer,
        model_type: ModelType,
    ) -> Self {
        Self {
            model,
            tokenizer,
            model_type,
        }
    }

    /// Retrieve a builder object for constructing a [`SentenceTransformer`] instance.
    pub fn builder() -> SentenceTransformerBuilder<Uninitialised> {
        SentenceTransformerBuilder::new()
    }

    /// Load a [`SentenceTransformer`] core from a folder containing the core, config, and tokenizer
    /// json files. The core should be saved in the SafeTensors format. Often, these folders
    /// are created by huggingface libraries when pulling a core from the hub, and are saved in
    /// the `~/.cache/huggingface/hub/models` directory.
    pub(crate) fn from_model_repo(
        model_repo_folder: &ModelRepo,
        device: &Device,
        pooling_strategy: Option<PoolingStrategy>,
    ) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "st-from-folder");
        let _enter = span.enter();

        let ModelRepoFiles {
            model_weights: model_weights_path,
            ..
        } = model_repo_folder.file_paths()?;

        let st_config =
            SentenceTransformerConfig::try_from_model_repo(model_repo_folder, pooling_strategy)?;

        let tokenizer_config_str = serde_json::to_string(&st_config.tokenizer_config)?;

        let mut tokenizer = Tokenizer::from_str(&tokenizer_config_str)?;

        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        let embedder_model =
            load_pretrained_model(model_weights_path, st_config.embedder_config, device)?;

        Ok(Self::new(embedder_model, tokenizer, st_config.model_type))
    }

    pub fn tokenize<'s, E>(&self, sentences: Vec<E>) -> Result<Vec<Encoding>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        Ok(self.tokenizer.encode_batch_fast(sentences, true)?)
    }

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

pub trait BuilderState {}

pub struct Uninitialised;
pub struct Initialised;

impl BuilderState for Uninitialised {}
impl BuilderState for Initialised {}

pub struct SentenceTransformerBuilder<S>
where
    S: BuilderState,
{
    model_repo: Option<ModelRepo>,
    pooling_strategy: Option<PoolingStrategy>,
    device: Device,
    _marker: PhantomData<S>,
}

impl Default for SentenceTransformerBuilder<Uninitialised> {
    fn default() -> Self {
        Self::new()
    }
}

impl SentenceTransformerBuilder<Uninitialised> {
    pub fn new() -> SentenceTransformerBuilder<Uninitialised> {
        Self {
            model_repo: None,
            pooling_strategy: None,
            device: Device::Cpu,
            _marker: PhantomData,
        }
    }
}

impl<S> SentenceTransformerBuilder<S>
where
    S: BuilderState,
{
    pub fn with_model_repo<MR: AsRef<str>>(
        self,
        model_repo: MR,
    ) -> Result<SentenceTransformerBuilder<Initialised>> {
        let (repo_id, revision) = utils::parse_repo_string(model_repo.as_ref())?;
        let repo = Repo::with_revision(repo_id.to_owned(), RepoType::Model, revision.to_owned());
        let api = Api::new()?;
        let api_repo = api.repo(repo);
        let model_repo = ModelRepo::from_api_repo(api_repo);
        Ok(SentenceTransformerBuilder::<Initialised> {
            model_repo: Some(model_repo),
            pooling_strategy: self.pooling_strategy,
            device: self.device,
            _marker: PhantomData,
        })
    }

    pub fn with_model_folder<MR: AsRef<Path>>(
        self,
        model_folder: MR,
    ) -> SentenceTransformerBuilder<Initialised> {
        let model_repo_folder = ModelRepo::from_path(model_folder.as_ref());
        SentenceTransformerBuilder::<Initialised> {
            model_repo: Some(model_repo_folder),
            pooling_strategy: self.pooling_strategy,
            device: self.device,
            _marker: PhantomData,
        }
    }

    pub fn with_pooling_strategy(self, pooling_strategy: PoolingStrategy) -> Self {
        Self {
            pooling_strategy: Some(pooling_strategy),
            ..self
        }
    }

    pub fn with_device(self, device: Device) -> Self {
        Self { device, ..self }
    }

    #[cfg(feature = "metal")]
    pub fn with_metal_device(self) -> Result<Self> {
        let device = Device::new_metal(0)?;

        Ok(Self { device, ..self })
    }

    #[cfg(feature = "cuda")]
    pub fn with_cuda_device(self) -> Result<Self> {
        let device = Device::new_cuda(0)?;

        Ok(Self { device, ..self })
    }
}

impl SentenceTransformerBuilder<Initialised> {
    pub fn build(self) -> Result<SentenceTransformer> {
        match self.model_repo {
            None => Err(Error::ModelLoad("No model directory or repository given.")),
            Some(mr) => {
                SentenceTransformer::from_model_repo(&mr, &self.device, self.pooling_strategy)
            }
        }
    }
}
