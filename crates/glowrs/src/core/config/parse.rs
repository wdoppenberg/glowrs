use std::fs;
use std::path::PathBuf;

use crate::core::config::model::{
    BaseModelConfig, EmbedderConfig, ModelType, SentenceTransformerConfig,
};
use crate::core::repo::{ModelRepo, ModelRepoFiles};
use crate::pooling::{PoolConfig, PoolingStrategy};
use crate::{Error, Result};

/// Parse the core configuration from the given core directory.
pub(crate) fn parse_config(
    // Directory containing all core files (in a HF repo)
    model_repo: &ModelRepo,
    // If not given, it'll be inferred from the core configuration
    pooling_strategy: Option<PoolingStrategy>,
) -> Result<SentenceTransformerConfig> {
    let ModelRepoFiles {
        config,
        tokenizer_config,
        pooling_config,
        ..
    } = model_repo.file_paths()?;

    // Parse config.json
    let config_str = &fs::read_to_string(config)?;
    let hf_config: BaseModelConfig = serde_json::from_str(config_str)?;
    let embedder_config: EmbedderConfig = serde_json::from_str(config_str)?;

    // Parse tokenizer.json
    let tokenizer_config: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(tokenizer_config)?)?;

    let model_type = get_backend_model_type(&hf_config, pooling_config, pooling_strategy)?;

    Ok(SentenceTransformerConfig {
        embedder_config,
        model_type,
        tokenizer_config,
    })
}

/// Get the backend core type from the given core configuration.
///
/// Source: `text-embeddings-inference`: [`backends/candle/src/lib.rs`](https://github.com/huggingface/text-embeddings-inference/blob/7e55c61c2a39612ade5db9b929ffc883913ae0f3/backends/candle/src/lib.rs)
pub(crate) fn get_backend_model_type(
    config: &BaseModelConfig,
    pooling_config_path: Option<PathBuf>,
    pooling: Option<PoolingStrategy>,
) -> Result<ModelType> {
    if let Some(p) = pooling {
        for arch in &config.architectures {
            if matches!(p, PoolingStrategy::Splade) && arch.ends_with("MaskedLM") {
                return Ok(ModelType::Embedding(PoolingStrategy::Splade));
            } else if arch.ends_with("Classification") {
                tracing::warn!(
                    "`--pooling` arg is set but core is a classifier. Ignoring `--pooling` arg."
                );
                return Ok(ModelType::Classifier);
            }
        }
    }

    if Some(PoolingStrategy::Splade) == pooling {
        return Err(Error::ModelLoad(
            "Splade pooling is not supported: core is not a *ForMaskedLM core",
        ));
    }

    // Set pooling
    let pool: Result<_> = match (pooling, pooling_config_path) {
        (Some(ps), _) => Ok(ps),
        (None, Some(pooling_config_path)) => {
            // Load pooling config

            let config = fs::read_to_string(pooling_config_path)?;
            // .map_err(|_| Err(Error::InvalidArgument("The `--pooling` arg is not set and we could not find a pooling configuration (`1_Pooling/config.json`) for this core.")))?;

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
        (_, _) => Err(Error::NoPoolingConfiguration(
            "No pooling configuration provided or found in model repository.",
        )),
    };
    Ok(ModelType::Embedding(pool?))
}

#[cfg(test)]
mod test {
    use super::*;
    use std::path::{Path, PathBuf};

    fn parse_config_helper(path: &Path, expected_model_type: ModelType) -> Result<()> {
        let model_repo = ModelRepo::from_path(path);

        let config = model_repo.get_config()?;
        assert_eq!(config.model_type, expected_model_type);
        Ok(())
    }

    #[test]
    fn test_parse_all_minilm_l6_v2_config() -> Result<()> {
        let model_root = PathBuf::from("tests/fixtures/all-MiniLM-L6-v2");
        parse_config_helper(
            model_root.as_path(),
            ModelType::Embedding(PoolingStrategy::Mean),
        )
    }

    // TODO: Make sure `bert-base-uncased` works
    // #[test]
    // fn test_parse_bert_base_uncased() -> Result<()> {
    //     let model_root = PathBuf::from("tests/fixtures/bert-base-uncased");
    //     parse_config_helper(
    //         model_root.as_path(),
    //         ModelType::Embedding(PoolingStrategy::Splade),
    //     )
    // }

    #[test]
    fn test_get_backend_model_type() {
        let config = BaseModelConfig {
            architectures: vec!["BertForMaskedLM".to_string()],
            model_type: "bert".to_string(),
            max_position_embeddings: 512,
            pad_token_id: 0,
            id2label: None,
            label2id: None,
        };
        let model_type =
            get_backend_model_type(&config, None, Some(PoolingStrategy::Mean)).unwrap();
        assert_eq!(model_type, ModelType::Embedding(PoolingStrategy::Mean));
    }
}
