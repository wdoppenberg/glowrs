use candle_core::{DType, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    bert::Config as BertConfig, distilbert::Config as DistilBertConfig,
    jina_bert::Config as JinaBertConfig,
};
use std::ops::Deref;
use std::path::Path;
use tokenizers::{EncodeInput, Tokenizer};

// Re-exports
pub use candle_transformers::models::{
    bert::BertModel, distilbert::DistilBertModel, jina_bert::BertModel as JinaBertModel,
};
use serde::Deserialize;

use crate::model::device::DEVICE;
use crate::model::pooling::{pool_embeddings, PoolingStrategy};
use crate::model::utils::normalize_l2;
use crate::{Error, Result, Usage};

#[cfg(test)]
use candle_nn::VarMap;

pub(crate) enum ModelConfig {
    Bert(BertConfig),
    JinaBert(JinaBertConfig),
    DistilBert(DistilBertConfig),
}

#[derive(Deserialize)]
struct BaseModelConfig {
    architectures: Option<Vec<String>>,
}

pub(crate) fn parse_config(config_str: &str) -> Result<ModelConfig> {
    use Error::*;
    let base_config: BaseModelConfig = serde_json::from_str(config_str)?;

    let config = match base_config.architectures {
        Some(arch) => {
            if arch.is_empty() {
                return Err(InvalidModelConfig("No architectures found"));
            }

            if arch.len() > 1 {
                return Err(InvalidModelConfig("Multiple architectures not supported"));
            }

            match arch.first().map(String::as_str) {
                Some("BertModel") => {
                    let config: BertConfig = serde_json::from_str(config_str)?;
                    ModelConfig::Bert(config)
                }
                Some("JinaBertForMaskedLM") => {
                    let config: JinaBertConfig = serde_json::from_str(config_str)?;
                    ModelConfig::JinaBert(config)
                }
                Some("DistilBertForMaskedLM") => {
                    let config: DistilBertConfig = serde_json::from_str(config_str)?;
                    ModelConfig::DistilBert(config)
                }
                _ => return Err(InvalidModelConfig("Invalid model architecture")),
            }
        }
        None => return Err(InvalidModelConfig("Model architecture not found")),
    };

    Ok(config)
}

pub(crate) fn load_model<T>(vb: VarBuilder, model_config: ModelConfig) -> Result<T>
where
    T: Deref<Target = dyn EmbedderModel> + From<Box<dyn EmbedderModel>> + AsRef<dyn EmbedderModel>,
{
    match model_config {
        ModelConfig::Bert(cfg) => Ok(T::from(Box::new(BertModel::load(vb, &cfg)?))),
        ModelConfig::JinaBert(cfg) => Ok(T::from(Box::new(JinaBertModel::new(vb, &cfg)?))),
        ModelConfig::DistilBert(cfg) => Ok(T::from(Box::new(DistilBertModel::load(vb, &cfg)?))),
    }
}

/// Load models.
pub(crate) fn load_pretrained_model<T>(model_path: &Path, config_path: &Path) -> Result<T>
where
    T: Deref<Target = dyn EmbedderModel> + From<Box<dyn EmbedderModel>> + AsRef<dyn EmbedderModel>,
{
    let config_str = std::fs::read_to_string(config_path)?;
    let model_config = parse_config(&config_str)?;

    // TODO: Make DType configurable
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &DEVICE)? };
    load_model::<T>(vb, model_config)
}

/// Trait for embedding models
pub trait EmbedderModel: Send + Sync {
    fn encode(&self, token_ids: &Tensor) -> Result<Tensor>;
}

impl EmbedderModel for BertModel {
    #[inline]
    fn encode(&self, token_ids: &Tensor) -> Result<Tensor> {
        let token_type_ids = token_ids.zeros_like()?;
        Ok(self.forward(token_ids, &token_type_ids)?)
    }
}

impl EmbedderModel for JinaBertModel {
    #[inline]
    fn encode(&self, token_ids: &Tensor) -> Result<Tensor> {
        Ok(self.forward(token_ids)?)
    }
}

impl EmbedderModel for DistilBertModel {
    #[inline]
    fn encode(&self, token_ids: &Tensor) -> Result<Tensor> {
        let attention_mask = token_ids.ones_like()?;
        Ok(self.forward(token_ids, &attention_mask)?)
    }
}

/// Encodes a batch of sentences by tokenizing them and running encoding them with the model,
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
pub(crate) fn encode_batch_with_usage<'s, E>(
    model: &dyn EmbedderModel,
    tokenizer: &Tokenizer,
    sentences: Vec<E>,
    pooling_strategy: &PoolingStrategy,
    normalize: bool,
) -> Result<(Tensor, Usage)>
where
    E: Into<EncodeInput<'s>> + Send,
{
    let tokens = tokenizer.encode_batch(sentences, true)?;

    let prompt_tokens = tokens.len() as u32;

    let usage = Usage {
        prompt_tokens,
        total_tokens: prompt_tokens,
    };

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Tensor::new(tokens.as_slice(), &DEVICE)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;

    tracing::trace!("running inference on batch {:?}", token_ids.shape());
    let embeddings = model.encode(&token_ids)?;
    tracing::trace!("generated embeddings {:?}", embeddings.shape());

    // Apply pooling
    let pooled_embeddings = pool_embeddings(&embeddings, pooling_strategy)?;

    // Normalize embeddings (if required)
    let embeddings = if normalize {
        normalize_l2(&pooled_embeddings)?
    } else {
        pooled_embeddings
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
pub(crate) fn encode_batch<'s, E>(
    model: &dyn EmbedderModel,
    tokenizer: &Tokenizer,
    sentences: Vec<E>,
    pooling_strategy: &PoolingStrategy,
    normalize: bool,
) -> Result<Tensor>
where
    E: Into<EncodeInput<'s>> + Send,
{
    let (out, _) =
        encode_batch_with_usage(model, tokenizer, sentences, pooling_strategy, normalize)?;
    Ok(out)
}

#[cfg(test)]
pub(crate) fn load_random_model<T>(model_config: ModelConfig) -> Result<T>
where
    T: Deref<Target = dyn EmbedderModel> + From<Box<dyn EmbedderModel>> + AsRef<dyn EmbedderModel>,
{
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &DEVICE);

    load_model::<T>(vb, model_config)
}

#[cfg(test)]
pub(crate) fn load_zeros_model<T>(model_config: ModelConfig) -> Result<T>
where
    T: Deref<Target = dyn EmbedderModel> + From<Box<dyn EmbedderModel>> + AsRef<dyn EmbedderModel>,
{
    // TODO: Make DType configurable
    let vb = VarBuilder::zeros(DType::F32, &DEVICE);
    load_model::<T>(vb, model_config)
}

#[cfg(test)]
mod test {
    use super::*;

    const BERT_CONFIG_PATH: &str = "tests/fixtures/all-MiniLM-L6-v2/config.json";
    const JINABERT_CONFIG_PATH: &str = "tests/fixtures/jina-embeddings-v2-base-en/config.json";
    const DISTILBERT_CONFIG_PATH: &str = "tests/fixtures/multi-qa-distilbert-dot-v1/config.json";

    #[test]
    fn test_parse_config_bert() -> Result<()> {
        let path = Path::new(BERT_CONFIG_PATH);
        let config_str = std::fs::read_to_string(path)?;

        let config = parse_config(&config_str)?;

        match config {
            ModelConfig::Bert(_) => {}
            _ => panic!("Invalid config type"),
        }

        Ok(())
    }

    #[test]
    fn test_parse_config_jinabert() -> Result<()> {
        let path = Path::new(JINABERT_CONFIG_PATH);

        let config_str = std::fs::read_to_string(path)?;

        let config = parse_config(&config_str)?;

        match config {
            ModelConfig::JinaBert(_) => {}
            _ => panic!("Invalid config type"),
        }

        Ok(())
    }

    #[test]
    fn test_parse_config_distilbert() -> Result<()> {
        let path = Path::new(DISTILBERT_CONFIG_PATH);

        let config_str = std::fs::read_to_string(path)?;

        let config = parse_config(&config_str)?;

        match config {
            ModelConfig::DistilBert(_) => {}
            _ => panic!("Invalid config type"),
        }

        Ok(())
    }

    #[test]
    fn test_forward_bert() -> Result<()> {
        let path = Path::new(BERT_CONFIG_PATH);

        let config_str = std::fs::read_to_string(path)?;
        let config = parse_config(&config_str)?;

        let model: Box<_> = load_random_model(config)?;

        let token_ids = Tensor::zeros(&[1, 128], DType::U32, &DEVICE)?;

        let embeddings = model.encode(&token_ids)?;

        let (_n_sentence, out_tokens, _hidden_size) = embeddings.dims3()?;

        assert_eq!(out_tokens, 128);

        Ok(())
    }

    #[test]
    fn test_forward_jinabert() -> Result<()> {
        let path = Path::new(JINABERT_CONFIG_PATH);

        let config_str = std::fs::read_to_string(path)?;
        let config = parse_config(&config_str)?;

        let model: Box<dyn EmbedderModel> = load_random_model(config)?;

        let token_ids = Tensor::zeros(&[1, 128], DType::U32, &DEVICE)?;

        let embeddings = model.encode(&token_ids)?;

        let (_n_sentence, out_tokens, _hidden_size) = embeddings.dims3()?;

        assert_eq!(out_tokens, 128);

        Ok(())
    }

    #[test]
    fn test_forward_distilbert() -> Result<()> {
        let path = Path::new(DISTILBERT_CONFIG_PATH);

        let config_str = std::fs::read_to_string(path)?;
        let config = parse_config(&config_str)?;

        let model: Box<dyn EmbedderModel> = load_random_model(config)?;

        let token_ids = Tensor::zeros(&[1, 128], DType::U32, &DEVICE)?;

        let embeddings = model.encode(&token_ids)?;

        let (_n_sentence, out_tokens, _hidden_size) = embeddings.dims3()?;

        assert_eq!(out_tokens, 128);

        Ok(())
    }
}
