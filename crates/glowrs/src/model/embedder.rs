use anyhow::{Context, Error, Result};
use candle_core::{Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    bert::Config as BertConfig, jina_bert::Config as JinaBertConfig,
};
use tokenizers::Tokenizer;

// Re-exports
pub use candle_transformers::models::{bert::BertModel, jina_bert::BertModel as JinaBertModel};
use serde::Deserialize;

use crate::model::device::DEVICE;
use crate::model::utils::normalize_l2;
use crate::{Sentences, Usage};

pub(crate) enum ModelConfig {
    Bert(BertConfig),
    JinaBert(JinaBertConfig),
}

#[derive(Deserialize)]
struct BaseModelConfig {
    architectures: Option<Vec<String>>,
}

pub(crate) fn parse_config(config_str: &str) -> Result<ModelConfig> {
    let base_config: BaseModelConfig = serde_json::from_str(config_str)?;

    let config = match base_config.architectures {
        Some(arch) => match arch.first().map(String::as_str) {
            Some("BertModel") => {
                let config: BertConfig = serde_json::from_str(config_str)?;
                ModelConfig::Bert(config)
            }
            Some("JinaBertForMaskedLM") => {
                let config: JinaBertConfig = serde_json::from_str(config_str)?;
                ModelConfig::JinaBert(config)
            }
            _ => return Err(Error::msg("Invalid model architecture")),
        },
        None => return Err(Error::msg("Model architecture not found")),
    };

    Ok(config)
}

/// Load models.
pub(crate) fn load_model(vb: VarBuilder, cfg: &ModelConfig) -> Result<Box<dyn EmbedderModel>> {
    match cfg {
        ModelConfig::Bert(cfg) => Ok(Box::new(BertModel::load(vb, cfg)?)),
        ModelConfig::JinaBert(cfg) => Ok(Box::new(JinaBertModel::new(vb, cfg)?)),
    }
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
    let embeddings = model.encode(&token_ids)?;
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

#[cfg(test)]
mod test {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_config_bert() -> Result<()> {
        let config = r#"
        {
            "_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
            "architectures": [
                "BertModel"
            ],
            "attention_probs_dropout_prob": 0.1,
            "classifier_dropout": null,
            "gradient_checkpointing": false,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 384,
            "initializer_range": 0.02,
            "intermediate_size": 1536,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "torch_dtype": "float32",
            "transformers_version": "4.36.2",
            "type_vocab_size": 2,
            "use_cache": true,
            "vocab_size": 30522
        }
        "#;

        let mut file = NamedTempFile::new()?;
        writeln!(file, "{}", config)?;

        let config_str = std::fs::read_to_string(file.path())?;

        let config = parse_config(&config_str)?;

        match config {
            ModelConfig::Bert(_) => {}
            _ => panic!("Invalid config type"),
        }

        Ok(())
    }

    #[test]
    fn test_parse_config_jinabert() -> Result<()> {
        let config = r#"
        {
            "_name_or_path": "jinaai/jina-bert-implementation",
            "model_max_length": 8192,
            "architectures": [
                "JinaBertForMaskedLM"
            ],
            "attention_probs_dropout_prob": 0.0,
            "auto_map": {
                "AutoConfig": "jinaai/jina-bert-implementation--configuration_bert.JinaBertConfig",
                "AutoModelForMaskedLM": "jinaai/jina-bert-implementation--modeling_bert.JinaBertForMaskedLM",
                "AutoModel": "jinaai/jina-bert-implementation--modeling_bert.JinaBertModel",
                "AutoModelForSequenceClassification": "jinaai/jina-bert-implementation--modeling_bert.JinaBertForSequenceClassification"
            },
            "classifier_dropout": null,
            "gradient_checkpointing": false,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 8192,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "position_embedding_type": "alibi",
            "torch_dtype": "float32",
            "transformers_version": "4.26.0",
            "type_vocab_size": 2,
            "use_cache": true,
            "vocab_size": 30528,
            "feed_forward_type": "geglu",
            "emb_pooler": "mean"
        }
        "#;

        let mut file = NamedTempFile::new()?;
        writeln!(file, "{}", config)?;

        let config_str = std::fs::read_to_string(file.path())?;

        let config = parse_config(&config_str)?;

        match config {
            ModelConfig::JinaBert(_) => {}
            _ => panic!("Invalid config type"),
        }

        Ok(())
    }
}
