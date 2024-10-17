use hf_hub::api::sync::ApiRepo;
use std::path::{Path, PathBuf};

use crate::core::config::model::SentenceTransformerConfig;
use crate::core::config::parse::parse_config;
use crate::{Error, Result};

/// Represents a folder with core weights structured as a repository on HF Hub.
pub enum ModelRepo {
    Folder(PathBuf),
    ApiRepo(Box<ApiRepo>),
}

const SAFETENSORS_FILE: &str = "model.safetensors";
const PTH_FILE: &str = "pytorch_model.bin";
const POOLING_CONFIG_FILE: &str = "1_Pooling/config.json";

impl ModelRepo {
    pub fn from_path<P>(root: P) -> Self
    where
        P: AsRef<Path>,
    {
        Self::Folder(root.as_ref().to_owned())
    }

    pub fn from_api_repo(api_repo: ApiRepo) -> Self {
        Self::ApiRepo(Box::new(api_repo))
    }

    /// Get the relevant repository files.
    ///
    /// **Warning**: Will download model weights if not present in the expected
    /// folder in the Huggingface cache.
    pub(crate) fn file_paths(&self) -> Result<ModelRepoFiles> {
        let root = match self {
            ModelRepo::Folder(pathbuf) => pathbuf.to_owned(),
            ModelRepo::ApiRepo(api_repo) => {
                let model_path = api_repo
                    .get(SAFETENSORS_FILE)
                    .or_else(|_e| api_repo.get(PTH_FILE))?;

                let _ = api_repo.get("config.json")?;

                let _ = api_repo.get("tokenizer.json")?;

                let pooling_dir_opt = api_repo.get("1_Pooling/config.json").ok();
                if pooling_dir_opt.is_none() {
                    tracing::info!(
                        "No pooling configuration found. Using default or given strategy."
                    );
                }

                let root = model_path
                    .parent()
                    .expect("Model path has no parent directory");

                root.to_owned()
            }
        };
        let config = root.join("config.json");
        let tokenizer_config = root.join("tokenizer.json");

        for p in [&config, &tokenizer_config] {
            if !p.exists() {
                return Err(Error::ModelLoad("Repository misses configuration files."));
            }
        }

        // Safetensors get precedence over pth.
        let model_weights = if root.join(SAFETENSORS_FILE).exists() {
            ModelWeightsPath::Safetensors(root.join(SAFETENSORS_FILE))
        } else if root.join(PTH_FILE).exists() {
            ModelWeightsPath::Pth(root.join(PTH_FILE))
        } else {
            return Err(Error::ModelLoad(
                "Repository doesn't contain model weights.",
            ));
        };

        let pooling_config = if root.join(POOLING_CONFIG_FILE).exists() {
            Some(root.join(POOLING_CONFIG_FILE))
        } else {
            None
        };

        Ok(ModelRepoFiles {
            config,
            tokenizer_config,
            model_weights,
            pooling_config,
        })
    }

    pub fn get_config(&self) -> Result<SentenceTransformerConfig> {
        parse_config(self, None)
    }
}

pub(crate) struct ModelRepoFiles {
    pub(crate) config: PathBuf,
    pub(crate) tokenizer_config: PathBuf,
    pub(crate) model_weights: ModelWeightsPath,
    pub(crate) pooling_config: Option<PathBuf>,
}

pub(crate) enum ModelWeightsPath {
    Pth(PathBuf),
    Safetensors(PathBuf),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_valid_model_repo() -> Result<()> {
        let dir = tempdir()?;
        let config_path = dir.path().join("config.json");
        let tokenizer_path = dir.path().join("tokenizer.json");
        let model_path = dir.path().join("model.safetensors");

        fs::write(&config_path, "{}")?;
        fs::write(&tokenizer_path, "{}")?;
        fs::write(&model_path, "{}")?;

        let repo = ModelRepo::from_path(dir.path());
        let repo_files = repo.file_paths();
        assert!(repo_files.is_ok());

        Ok(())
    }

    #[test]
    fn test_invalid_model_repo_missing_files() -> Result<()> {
        let dir = tempdir()?;
        let config_path = dir.path().join("config.json");
        let tokenizer_path = dir.path().join("tokenizer.json");

        fs::write(&config_path, "{}")?;
        fs::write(&tokenizer_path, "{}")?;

        let repo = ModelRepo::from_path(dir.path());
        let repo_files = repo.file_paths();
        assert!(repo_files.is_err());

        Ok(())
    }

    #[test]
    fn test_model_repo_with_pooling_config() -> Result<()> {
        let dir = tempdir()?;
        let config_path = dir.path().join("config.json");
        let tokenizer_path = dir.path().join("tokenizer.json");
        let model_path = dir.path().join("model.safetensors");
        let pooling_config_path = dir.path().join("1_Pooling/config.json");

        fs::create_dir_all(pooling_config_path.parent().unwrap())?;
        fs::write(&config_path, "{}")?;
        fs::write(&tokenizer_path, "{}")?;
        fs::write(&model_path, "{}")?;
        fs::write(&pooling_config_path, "{}")?;

        let repo = ModelRepo::from_path(dir.path());
        let repo_files = repo.file_paths();
        assert!(repo_files.is_ok());

        Ok(())
    }

    #[test]
    fn test_model_repo_with_pt_weights() -> Result<()> {
        let dir = tempdir()?;
        let config_path = dir.path().join("config.json");
        let tokenizer_path = dir.path().join("tokenizer.json");
        let model_path = dir.path().join("pytorch_model.bin");

        fs::write(&config_path, "{}")?;
        fs::write(&tokenizer_path, "{}")?;
        fs::write(&model_path, r"\b")?;

        let repo = ModelRepo::from_path(dir.path());
        let ModelRepoFiles { model_weights, .. } = repo.file_paths()?;
        assert!(matches!(model_weights, ModelWeightsPath::Pth(_)));

        Ok(())
    }
}
