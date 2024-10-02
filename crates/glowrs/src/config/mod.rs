pub(crate) mod model;
pub(crate) mod parse;

#[cfg(test)]
mod tests {
    const BERT_CONFIG_PATH: &str = "tests/fixtures/all-MiniLM-L6-v2";
    const JINABERT_CONFIG_PATH: &str = "tests/fixtures/jina-embeddings-v2-base-en";
    const DISTILBERT_CONFIG_PATH: &str = "tests/fixtures/multi-qa-distilbert-dot-v1";

    use std::path::Path;

    use super::*;
    use crate::config::parse::parse_config;
    use crate::pooling::PoolingStrategy;
    use crate::Result;

    fn test_parse_config_helper(config_path: &str, expected_type: model::ModelType) -> Result<()> {
        let path = Path::new(config_path);

        let config = parse_config(path, None)?;

        assert_eq!(config.model_type, expected_type);

        Ok(())
    }

    #[test]
    fn test_parse_config_bert() -> Result<()> {
        test_parse_config_helper(
            BERT_CONFIG_PATH,
            crate::config::model::ModelType::Embedding(PoolingStrategy::Mean),
        )
    }

    #[test]
    fn test_parse_config_jinabert() -> Result<()> {
        test_parse_config_helper(
            JINABERT_CONFIG_PATH,
            crate::config::model::ModelType::Embedding(PoolingStrategy::Mean),
        )
    }

    #[test]
    fn test_parse_config_distilbert() -> Result<()> {
        test_parse_config_helper(
            DISTILBERT_CONFIG_PATH,
            crate::config::model::ModelType::Embedding(PoolingStrategy::Cls),
        )
    }
}
