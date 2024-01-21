use candle_core::{Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    bert::BertModel,
    jina_bert::{BertModel as _JinaBertModel, Config as JinaBertConfig},
};

/// The `SBert` trait represents a semantic embedding model based on the SBert architecture.
pub trait SBert: Module + Sized {
    type Config;
    const MODEL_REPO_NAME: &'static str;
    const TOKENIZER_REPO_NAME: &'static str;

    /// Creates a new instance using the specified `VarBuilder` and configuration in accordance with default
    /// `candle` model initialization.
    fn new(vb: VarBuilder, cfg: &Self::Config) -> candle_core::Result<Self>;

    /// Returns the default configuration for the current implementation.
    fn default_config() -> Self::Config;
}

pub struct JinaBertBaseV2(_JinaBertModel);

impl Module for JinaBertBaseV2 {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.0.forward(xs)
    }
}

impl SBert for JinaBertBaseV2 {
    type Config = JinaBertConfig;
    const MODEL_REPO_NAME: &'static str = "jinaai/jina-embeddings-v2-base-en";
    const TOKENIZER_REPO_NAME: &'static str = "sentence-transformers/all-MiniLM-L6-v2";

    fn new(vb: VarBuilder, cfg: &Self::Config) -> candle_core::Result<Self> {
        let model = _JinaBertModel::new(vb, cfg)?;
        Ok(Self(model))
    }

    fn default_config() -> Self::Config {
        Self::Config::v2_base()
    }
}

pub struct AllMiniLmL6V2(BertModel);

// impl Module for AllMiniLmL6V2 {
// 	fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
// 		self.0.forward(xs)
// 	}
// }