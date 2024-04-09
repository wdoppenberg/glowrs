use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct EmbeddingsConfig {
	pub model_repo: String,
}

#[derive(Debug, Deserialize)]
pub struct Config {
	pub models: Vec<EmbeddingsConfig>,
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_config_deserialization() {
		let config = r#"
		{
			"models": [
				{
					"model_repo": "test_repo:test_revision"
				}
			]
		}
		"#;

		let config: Config = serde_json::from_str(config).unwrap();
		assert_eq!(config.models.len(), 1);
		assert_eq!(config.models[0].model_repo, "test_repo:test_revision");
	}
}