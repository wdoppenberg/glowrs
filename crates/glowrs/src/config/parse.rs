use std::fs;
use std::path::Path;

use crate::config::model::{HFConfig, ModelConfig, ModelDefinition, ModelType};
use crate::pooling::{PoolConfig, PoolingStrategy};
use crate::{Error, Result};

/// Parse the model configuration from the given model directory.
pub(crate) fn parse_config(
    // Directory containing all model files (in a HF repo)
    model_root: &Path,
    // If not given, it'll be inferred from the model configuration
    pooling_strategy: Option<PoolingStrategy>,
) -> Result<ModelDefinition> {
    let config_path = model_root.join("config.json");
    let config_str = &fs::read_to_string(config_path)?;
    let hf_config: HFConfig = serde_json::from_str(config_str)?;
    let model_config: ModelConfig = serde_json::from_str(config_str)?;

    let model_type = get_backend_model_type(&hf_config, model_root, pooling_strategy)?;

    Ok(ModelDefinition {
        model_config,
        model_type,
    })
}

/// Get the backend model type from the given model configuration.
///
/// Source: `text-embeddings-inference`: [`backends/candle/src/lib.rs`](https://github.com/huggingface/text-embeddings-inference/blob/7e55c61c2a39612ade5db9b929ffc883913ae0f3/backends/candle/src/lib.rs)
fn get_backend_model_type(
    config: &HFConfig,
    model_root: &Path,
    pooling: Option<PoolingStrategy>,
) -> Result<ModelType> {
    if let Some(p) = pooling.clone() {
        for arch in &config.architectures {
            if matches!(p, PoolingStrategy::Splade) && arch.ends_with("MaskedLM") {
                return Ok(ModelType::Embedding(PoolingStrategy::Splade));
            } else if arch.ends_with("Classification") {
                tracing::warn!(
                    "`--pooling` arg is set but model is a classifier. Ignoring `--pooling` arg."
                );
                return Ok(ModelType::Classifier);
            }
        }
    }

    if Some(PoolingStrategy::Splade) == pooling.clone() {
        return Err(Error::ModelLoad(
            "Splade pooling is not supported: model is not a *ForMaskedLM model",
        ));
    }

    // Set pooling
    let pool: Result<_> = match pooling {
        Some(ps) => Ok(ps),
        None => {
            // Load pooling config
            let config_path = model_root.join("1_Pooling/config.json");

            let config = fs::read_to_string(config_path)?;
            // .map_err(|_| Err(Error::InvalidArgument("The `--pooling` arg is not set and we could not find a pooling configuration (`1_Pooling/config.json`) for this model.")))?;

            let config: PoolConfig = serde_json::from_str(&config)?;
            // .map_err(|_| Err(Error::InvalidArgument("Failed to parse `1_Pooling/config.json`")).into())?;

            if config.pooling_mode_cls_token {
                Ok(PoolingStrategy::Cls)
            } else if config.pooling_mode_mean_tokens {
                Ok(PoolingStrategy::Mean)
            } else {
                return Err(Error::ModelLoad(
                    "Pooling config {config:?} is not supported",
                ));
            }
        }
    };
    Ok(ModelType::Embedding(pool?))
}

#[cfg(test)]
mod test {
    use super::*;
    use std::path::PathBuf;

    fn parse_config_helper(
        path: &Path,
        pooling_strategy: Option<PoolingStrategy>,
        expected_model_type: ModelType,
    ) -> Result<()> {
        let model_definition = parse_config(&path, pooling_strategy).unwrap();
        assert_eq!(model_definition.model_type, expected_model_type);
        Ok(())
    }

    #[test]
    fn test_parse_all_minilm_l6_v2_config() -> Result<()> {
        let model_root = PathBuf::from("tests/fixtures/all-MiniLM-L6-v2");
        parse_config_helper(
            model_root.as_path(),
            None,
            ModelType::Embedding(PoolingStrategy::Mean),
        )
    }

    #[test]
    fn test_parse_bert_base_uncased() -> Result<()> {
        let model_root = PathBuf::from("tests/fixtures/bert-base-uncased");
        parse_config_helper(
            model_root.as_path(),
            Some(PoolingStrategy::Splade),
            ModelType::Embedding(PoolingStrategy::Splade),
        )
    }

    #[test]
    fn test_get_backend_model_type() {
        let config = HFConfig {
            architectures: vec!["BertForMaskedLM".to_string()],
            model_type: "bert".to_string(),
            max_position_embeddings: 512,
            pad_token_id: 0,
            id2label: None,
            label2id: None,
        };
        let model_root = PathBuf::from("tests/fixtures/all-MiniLM-L6-v2/");
        let model_type =
            get_backend_model_type(&config, &model_root, Some(PoolingStrategy::Mean)).unwrap();
        assert_eq!(model_type, ModelType::Embedding(PoolingStrategy::Mean));
    }
}
