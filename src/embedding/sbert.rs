use std::string::ToString;
use candle_transformers::models::jina_bert::{BertModel as _JinaBertModel, Config as JinaBertConfig};
use candle_core::{Module, Tensor};
use candle_nn::VarBuilder;

/// The `SBert` trait represents a semantic embedding model based on the SBert architecture.
pub trait SBert: Module + Sized {
	type Config;

	/// Creates a new instance using the specified `VarBuilder` and configuration in accordance with default
	/// `candle` model initialization.
	fn new(vb: VarBuilder, cfg: &Self::Config) -> candle_core::Result<Self>;

	/// Returns the default configuration for the current implementation.
	fn default_config() -> Self::Config;

	/// Returns the name of the model repository on HF (e.g. `sentence-transformers/all-MiniLM-L6-v2`) as a string.
	fn model_repo_name() -> String;

	/// Returns the name of the repository for the tokenizer.
	fn tokenizer_repo_name() -> String;
}

pub struct JinaBertBaseV2(_JinaBertModel);

impl Module for JinaBertBaseV2 {
	fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
		self.0.forward(xs)
	}
}

impl SBert for JinaBertBaseV2 {
	type Config = JinaBertConfig;

	fn new(vb: VarBuilder, cfg: &Self::Config) -> candle_core::Result<Self> {
		let model = _JinaBertModel::new(vb, cfg)?;
		Ok(Self(model))
	}

	fn default_config() -> Self::Config {
		Self::Config::v2_base()
	}

	fn model_repo_name() -> String {
		"jinaai/jina-embeddings-v2-base-en".to_string()
	}

	fn tokenizer_repo_name() -> String {
		"sentence-transformers/all-MiniLM-L6-v2".to_string()
	}
}
