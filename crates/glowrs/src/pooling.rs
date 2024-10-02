use serde::Deserialize;

#[cfg(feature = "clap")]
use clap::ValueEnum;

/// Pooling strategy.
///
/// Source: `text-embeddings-inference`: [`backends/candle/src/lib.rs`](https://github.com/huggingface/text-embeddings-inference/blob/7e55c61c2a39612ade5db9b929ffc883913ae0f3/backends/candle/src/lib.rs)
#[cfg_attr(feature = "clap", derive(ValueEnum))]
#[derive(Debug, PartialEq, Clone)]
pub enum PoolingStrategy {
    /// Select the CLS token as embedding
    Cls,
    /// Apply Mean pooling to the model embeddings
    Mean,
    /// Apply SPLADE (Sparse Lexical and Expansion) to the model embeddings.
    /// This option is only available if the loaded model is a `ForMaskedLM` Transformer
    /// model.
    Splade,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PoolConfig {
    pub(crate) pooling_mode_cls_token: bool,
    pub(crate) pooling_mode_mean_tokens: bool,
    pooling_mode_max_tokens: bool,
    pooling_mode_mean_sqrt_len_tokens: bool,
}
