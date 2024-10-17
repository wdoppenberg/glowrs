//! Embedding core configuration data
//!
//! An embedder is defined by its core configuration (defined in the `config.json` in the root
//! of a Hugging Face core repository) the core type, and the pooling strategy (optionally
//! defined in a `1_Pooling/config.json` file in the core repository).

use crate::core::config::parse::parse_config;
use crate::core::repo::ModelRepo;
use crate::pooling::PoolingStrategy;
use crate::Result;
use candle_transformers::models::bert::Config as _BertConfig;
use candle_transformers::models::distilbert::Config as DistilBertConfig;
use candle_transformers::models::jina_bert::Config as _JinaBertConfig;
use serde::Deserialize;
use std::collections::HashMap;

/// The base HF embedding core configuration.
///
/// This represents the base fields present in a `config.json` for an embedding core.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct BaseModelConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    #[serde(alias = "n_positions")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub pad_token_id: usize,
    pub id2label: Option<HashMap<usize, String>>,
    pub label2id: Option<HashMap<String, usize>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum BertConfig {
    Bert(_BertConfig),
    JinaBert(_JinaBertConfig),
}

/// The given core type.
///
/// Based on the `model_type` key in the `config.json`, the given variant enables the parser
/// to know what specific core configuration data core to use when deserializing the non-base
/// keys.
#[derive(Deserialize)]
#[serde(tag = "model_type", rename_all = "kebab-case")]
pub(crate) enum EmbedderConfig {
    Bert(BertConfig),
    // XlmRoberta(BertConfig),
    // Camembert(BertConfig),
    // Roberta(BertConfig),
    #[serde(rename(deserialize = "distilbert"))]
    DistilBert(DistilBertConfig),
}

/// The embedding strategy used by a given core.
#[derive(Debug, PartialEq, Clone)]
pub enum ModelType {
    Classifier,
    Embedding(PoolingStrategy),
}

/// The core definition
pub struct SentenceTransformerConfig {
    pub(crate) embedder_config: EmbedderConfig,
    pub(crate) model_type: ModelType,
    pub(crate) tokenizer_config: serde_json::Value,
}

impl SentenceTransformerConfig {
    pub(crate) fn try_from_model_repo(
        model_repo: &ModelRepo,
        pooling_strategy: Option<PoolingStrategy>,
    ) -> Result<Self> {
        parse_config(model_repo, pooling_strategy)
    }
}
