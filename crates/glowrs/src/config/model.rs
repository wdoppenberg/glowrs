//! Embedding model configuration data model
//!
//! An embedder is defined by its model configuration (defined in the `config.json` in the root
//! of a Hugging Face model repository) the model type, and the pooling strategy (optionally
//! defined in a `1_Pooling/config.json` file in the model repository).

use candle_transformers::models::bert::Config as _BertConfig;
use candle_transformers::models::distilbert::Config as DistilBertConfig;
use candle_transformers::models::jina_bert::Config as _JinaBertConfig;
use serde::Deserialize;
use std::collections::HashMap;

use crate::pooling::PoolingStrategy;

/// The base HF embedding model configuration.
///
/// This represents the base fields present in a `config.json` for an embedding model.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct HFConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    #[serde(alias = "n_positions")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub pad_token_id: usize,
    pub id2label: Option<HashMap<String, String>>,
    pub label2id: Option<HashMap<String, usize>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum BertConfig {
    Bert(_BertConfig),
    JinaBert(_JinaBertConfig),
}

/// The given model type.
///
/// Based on the `model_type` key in the `config.json`, the given variant enables the parser
/// to know what specific model configuration data model to use when deserializing the non-base
/// keys.
#[derive(Deserialize)]
#[serde(tag = "model_type", rename_all = "kebab-case")]
pub(crate) enum ModelConfig {
    Bert(BertConfig),
    // XlmRoberta(BertConfig),
    // Camembert(BertConfig),
    // Roberta(BertConfig),
    #[serde(rename(deserialize = "distilbert"))]
    DistilBert(DistilBertConfig),
}

/// The embedding strategy used by a given model.
#[derive(Debug, PartialEq, Clone)]
pub enum ModelType {
    Classifier,
    Embedding(PoolingStrategy),
}

/// The model definition
pub(crate) struct ModelDefinition {
    pub(crate) model_config: ModelConfig,
    pub(crate) model_type: ModelType,
}
