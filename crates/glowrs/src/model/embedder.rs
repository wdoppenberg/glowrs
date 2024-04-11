use anyhow::{Context, Error, Result};
use candle_core::{DType, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    bert::Config as BertConfig, jina_bert::Config as JinaBertConfig,
};
use serde::de::DeserializeOwned;
use tokenizers::Tokenizer;

// Re-exports
pub use candle_transformers::models::{bert::BertModel, jina_bert::BertModel as JinaBertModel};

use crate::model::device::DEVICE;
use crate::model::utils::normalize_l2;
use crate::{Sentences, Usage};

/// Trait for loadable models.
pub trait LoadableModel: Sized {
    type Config: DeserializeOwned;
    fn load_model(vb: VarBuilder, cfg: &Self::Config) -> Result<Box<dyn EmbedderModel>>;

    fn empty_model() -> Result<Box<dyn EmbedderModel>>;
}

/// Trait for embedding models
pub trait EmbedderModel: Send + Sync {
    fn inner_forward(&self, token_ids: &Tensor) -> Result<Tensor>;
}

impl LoadableModel for BertModel {
    type Config = BertConfig;
    fn load_model(vb: VarBuilder, cfg: &Self::Config) -> Result<Box<dyn EmbedderModel>> {
        Ok(Box::new(Self::load(vb, cfg)?))
    }

    fn empty_model() -> Result<Box<dyn EmbedderModel>> {
        let vb = VarBuilder::zeros(DType::F32, &DEVICE);
        let cfg = Self::Config::default();
        Self::load_model(vb, &cfg)
    }
}

impl LoadableModel for JinaBertModel {
    type Config = JinaBertConfig;
    fn load_model(vb: VarBuilder, cfg: &Self::Config) -> Result<Box<dyn EmbedderModel>> {
        Ok(Box::new(Self::new(vb, cfg)?))
    }

    fn empty_model() -> Result<Box<dyn EmbedderModel>> {
        let vb = VarBuilder::zeros(DType::F32, &DEVICE);
        let cfg = Self::Config::v2_base();
        Self::load_model(vb, &cfg)
    }
}

impl EmbedderModel for BertModel {
    #[inline]
    fn inner_forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let token_type_ids = token_ids.zeros_like()?;
        Ok(self.forward(token_ids, &token_type_ids)?)
    }
}

impl EmbedderModel for JinaBertModel {
    #[inline]
    fn inner_forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        Ok(self.forward(token_ids)?)
    }
}

/// Represents the type of embedding algorithm used by the embedding model.
#[derive(Debug, PartialEq)]
pub enum EmbedderType {
    Bert,
    JinaBert,
}

/// Encodes a batch of sentences using an embedder model and a tokenizer,
/// and returns the embeddings along with the usage statistics.
///
/// # Arguments
///
/// * `model` - A reference to a `dyn EmbedderModel` trait object.
/// * `tokenizer` - A reference to a `Tokenizer`.
/// * `sentences` - A collection of sentences to encode.
/// * `normalize` - A boolean flag indicating whether to normalize the embeddings or not.
///
/// # Returns
///
/// Returns a tuple containing the embeddings as a `Tensor` and the usage statistics as a `Usage` struct.
///
/// # Errors
///
/// Returns an error if there is any failure during the encoding process.
///
pub(crate) fn encode_batch_with_usage(
    model: &dyn EmbedderModel,
    tokenizer: &Tokenizer,
    sentences: impl Into<Vec<String>>,
    normalize: bool,
) -> Result<(Tensor, Usage)> {
    let tokens = tokenizer
        .encode_batch(sentences.into(), true)
        .map_err(Error::msg)
        .context("Failed to encode batch.")?;

    let prompt_tokens = tokens.len() as u32;

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Tensor::new(tokens.as_slice(), &DEVICE)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;

    tracing::trace!("running inference on batch {:?}", token_ids.shape());
    let embeddings = model.inner_forward(&token_ids)?;
    tracing::trace!("generated embeddings {:?}", embeddings.shape());

    // Apply some avg-pooling by taking the mean model value for all tokens (including padding)
    let (_n_sentence, out_tokens, _hidden_size) = embeddings.dims3()?;
    let embeddings = (embeddings.sum(1)? / (out_tokens as f64))?;
    let embeddings = if normalize {
        normalize_l2(&embeddings)?
    } else {
        embeddings
    };

    // TODO: Incorrect usage calculation - fix
    let usage = Usage {
        prompt_tokens,
        total_tokens: prompt_tokens + (out_tokens as u32),
    };
    Ok((embeddings, usage))
}

/// Encodes a batch of sentences using the given `model` and `tokenizer`.
///
/// # Arguments
/// * `model` - A reference to the embedding model to use.
/// * `tokenizer` - A reference to the tokenizer to use.
/// * `sentences` - The sentences to encode.
/// * `normalize` - A flag indicating whether to normalize the embeddings.
///
/// # Returns
/// * `Result<Tensor>` - A result containing the encoded batch of sentences.
pub(crate) fn encode_batch(
    model: &dyn EmbedderModel,
    tokenizer: &Tokenizer,
    sentences: Sentences,
    normalize: bool,
) -> Result<Tensor> {
    let (out, _) = encode_batch_with_usage(model, tokenizer, sentences, normalize)?;
    Ok(out)
}
