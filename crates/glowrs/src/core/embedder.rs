use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::VarBuilder;

use tokenizers::{EncodeInput, Tokenizer};

// Re-exports
pub use candle_transformers::models::{
    bert::BertModel, distilbert::DistilBertModel, jina_bert::BertModel as JinaBertModel,
};

use crate::core::config::model::{BertConfig, EmbedderConfig, ModelType};
use crate::core::repo::ModelWeightsPath;
use crate::core::utils::normalize_l2;
use crate::pooling::PoolingStrategy;
use crate::{Result, Usage};

pub(crate) fn load_model(
    vb: VarBuilder,
    model_config: EmbedderConfig,
) -> Result<Box<dyn EmbedderModel>>
where
{
    match model_config {
        EmbedderConfig::Bert(cfg) => Ok(match cfg {
            BertConfig::Bert(cfg_inner) => Box::new(BertModel::load(vb, &cfg_inner)?),
            BertConfig::JinaBert(cfg_inner) => Box::new(JinaBertModel::new(vb, &cfg_inner)?),
        }),
        EmbedderConfig::DistilBert(cfg) => Ok(Box::new(DistilBertModel::load(vb, &cfg)?)),
    }
}

pub(crate) fn load_pretrained_model(
    model_weights_path: ModelWeightsPath,
    model_config: EmbedderConfig,
    device: &Device,
) -> Result<Box<dyn EmbedderModel>> {
    let vb = match model_weights_path {
        ModelWeightsPath::Pth(path) => VarBuilder::from_pth(&path, DType::F32, device)?,
        ModelWeightsPath::Safetensors(path) => unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)?
        },
    };

    load_model(vb, model_config)
}

/// Trait for embedder models
pub trait EmbedderModel: Send + Sync {
    fn encode(&self, token_ids: &Tensor) -> Result<Tensor>;

    #[inline]
    fn encode_with_pooling(
        &self,
        token_ids: &Tensor,
        pool_fn: fn(&Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let embeddings = &self.encode(token_ids)?;

        pool_fn(embeddings)
    }

    fn get_device(&self) -> &Device;
}

impl EmbedderModel for BertModel {
    #[inline]
    fn encode(&self, token_ids: &Tensor) -> Result<Tensor> {
        let token_type_ids = token_ids.zeros_like()?;
        Ok(self.forward(token_ids, &token_type_ids)?)
    }

    fn get_device(&self) -> &Device {
        &self.device
    }
}

impl EmbedderModel for JinaBertModel {
    #[inline]
    fn encode(&self, token_ids: &Tensor) -> Result<Tensor> {
        Ok(self.forward(token_ids)?)
    }

    fn get_device(&self) -> &Device {
        &self.device
    }
}

impl EmbedderModel for DistilBertModel {
    #[inline]
    fn encode(&self, token_ids: &Tensor) -> Result<Tensor> {
        let size = token_ids.dim(0)?;

        let mask: Vec<_> = (0..size)
            .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
            .collect();

        let mask = Tensor::from_slice(&mask, (size, size), token_ids.device())?;

        Ok(self.forward(token_ids, &mask)?)
    }

    fn get_device(&self) -> &Device {
        &self.device
    }
}

#[derive(Debug)]
pub struct EmbedOutput {
    pub embeddings: Tensor,
    pub usage: Usage,
}

/// Encodes a batch of sentences by tokenizing them and running encoding them with the core,
/// and returns the embeddings along with the usage statistics.
///
/// # Arguments
///
/// * `core` - A reference to a `dyn EmbedderModel` trait object.
/// * `tokenizer` - A reference to a `Tokenizer`.
/// * `sentences` - A collection of sentences to encode.
/// * `normalize` - A boolean flag indicating whether to normalize the embeddings or not.
///
/// # Returns
///
/// Returns a tuple containing the embeddings as a `Tensor` and the usage statistics as a `Usage` struct.
///
/// # Errors
///
/// Returns an error if there is any failure during the encoding process.
///
pub(crate) fn encode_batch_with_usage<'s, E>(
    model: &dyn EmbedderModel,
    tokenizer: &Tokenizer,
    sentences: Vec<E>,
    model_type: &ModelType,
    normalize: bool,
) -> Result<EmbedOutput>
where
    E: Into<EncodeInput<'s>> + Send,
{
    let tokens = tokenizer.encode_batch_fast(sentences, true)?;

    let prompt_tokens = tokens.len() as u32;

    let usage = Usage {
        prompt_tokens,
        total_tokens: prompt_tokens,
    };

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();

            Tensor::new(tokens.as_slice(), model.get_device())
        })
        .collect::<candle_core::Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;

    tracing::trace!("running inference on batch {:?}", token_ids.shape());

    // let embeddings = core.encode(&token_ids)?;
    let embeddings = model.encode(&token_ids)?;

    let pooling_strategy = match model_type {
        ModelType::Classifier => &PoolingStrategy::Cls, // TODO: Is this correct?
        ModelType::Embedding(ps) => ps,
    };

    let embeddings = match pooling_strategy {
        PoolingStrategy::Cls => embeddings.i((.., 0))?,
        PoolingStrategy::Mean => {
            let pad_id = tokenizer.get_padding().map_or(0, |pp| pp.pad_id);

            let attention_mask = token_ids
                .ne(pad_id)?
                .unsqueeze(D::Minus1)?
                .to_dtype(embeddings.dtype())?;

            embeddings.broadcast_mul(&attention_mask)?.sum(1)?
        }
        PoolingStrategy::Splade => panic!("SPLADE is not yet implemented."),
    };

    // Normalize embeddings (if required)
    let embeddings = {
        if normalize {
            normalize_l2(&embeddings)?
        } else {
            embeddings
        }
    };

    tracing::trace!("generated embeddings {:?}", embeddings.shape());
    Ok(EmbedOutput { embeddings, usage })
}

/// Encodes a batch of sentences using the given `core` and `tokenizer`.
///
/// # Arguments
/// * `core` - A reference to the embedding core to use.
/// * `tokenizer` - A reference to the tokenizer to use.
/// * `sentences` - The sentences to encode.
/// * `normalize` - A flag indicating whether to normalize the embeddings.
///
/// # Returns
/// * `Result<Tensor>` - A result containing the encoded batch of sentences.
pub(crate) fn encode_batch<'s, E>(
    model: &dyn EmbedderModel,
    tokenizer: &Tokenizer,
    sentences: Vec<E>,
    model_type: &ModelType,
    normalize: bool,
) -> Result<Tensor>
where
    E: Into<EncodeInput<'s>> + Send,
{
    let embed_output = encode_batch_with_usage(model, tokenizer, sentences, model_type, normalize)?;

    Ok(embed_output.embeddings)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::repo::ModelRepo;
    use std::path::Path;

    const BERT_PATH: &str = "tests/fixtures/all-MiniLM-L6-v2/";
    const JINABERT_PATH: &str = "tests/fixtures/jina-embeddings-v2-base-en/";
    const DISTILBERT_PATH: &str = "tests/fixtures/multi-qa-distilbert-dot-v1/";

    #[test]
    fn test_parse_config_bert() -> Result<()> {
        let path = Path::new(BERT_PATH);

        let model_repo = ModelRepo::from_path(path);

        let config = model_repo.get_config()?;

        match config.model_type {
            ModelType::Embedding(_) => {}
            _ => panic!("Invalid config type"),
        }

        Ok(())
    }

    #[test]
    fn test_parse_config_jinabert() -> Result<()> {
        let path = Path::new(JINABERT_PATH);

        let model_repo = ModelRepo::from_path(path);

        let config = model_repo.get_config()?;

        match config.model_type {
            ModelType::Embedding(_) => {}
            _ => panic!("Invalid config type"),
        }

        Ok(())
    }

    #[test]
    fn test_parse_config_distilbert() -> Result<()> {
        let path = Path::new(DISTILBERT_PATH);

        let model_repo = ModelRepo::from_path(path);

        let config = model_repo.get_config()?;

        match config.model_type {
            ModelType::Embedding(ps) => match ps {
                PoolingStrategy::Cls => {}
                _ => panic!("Invalid pooling type"),
            },
            _ => panic!("Invalid core type"),
        }

        Ok(())
    }
}
