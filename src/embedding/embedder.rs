use candle_core::{DType, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{
	bert::{BertModel, Config as BertConfig},
	jina_bert::{BertModel as _JinaBertModel, Config as JinaBertConfig},
};
use anyhow::{anyhow, Error, Result};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;
use crate::routes::{Sentences, Usage};
use crate::utils::device::DEVICE;
use crate::utils::normalize_l2;

/// This trait represents a semantic embedding model based on the Sentence-Bert architecture.
pub trait Embedder: Send {
	fn forward(&self, xs: &Tensor) -> Result<Tensor>;
	fn encode_batch_with_usage(
		&self,
		sentences: Sentences,
		normalize: bool,
		tokenizer: &Tokenizer,
	) -> Result<(Tensor, Usage)> {
		let s_vec: Vec<String> = sentences.into();
		let tokens = tokenizer
			.encode_batch(s_vec, true)
			.map_err(Error::msg)?;

		let prompt_tokens = tokens.len() as u32;

		let token_ids = tokens
			.iter()
			.map(|tokens| {
				let tokens = tokens.get_ids().to_vec();
				Tensor::new(tokens.as_slice(), &DEVICE)
			})
			.collect::<candle_core::Result<Vec<_>>>()?;

		let token_ids = Tensor::stack(&token_ids, 0)?;

		tracing::trace!("running inference on batch {:?}", token_ids.shape());
		let embeddings = self.forward(&token_ids)?;
		tracing::trace!("generated embeddings {:?}", embeddings.shape());

		// Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
		let (_n_sentence, out_tokens, _hidden_size) = embeddings.dims3()?;
		let embeddings = (embeddings.sum(1)? / (out_tokens as f64))?;
		let embeddings = if normalize {
			normalize_l2(&embeddings)?
		} else {
			embeddings
		};

		let usage = Usage {
			prompt_tokens,
			total_tokens: prompt_tokens + (out_tokens as u32),
		};
		Ok((embeddings, usage))
	}

	fn encode_batch(
		&self,
		sentences: Sentences,
		normalize: bool,
		tokenizer: &Tokenizer
	) -> Result<Tensor> {
		let (out, _) = self.encode_batch_with_usage(sentences, normalize, tokenizer)?;
		Ok(out)
	}

}

pub trait StaticLoadEmbedder: Sized + Embedder {
	fn new() -> Result<Self>;
}

pub trait DynLoadEmbedder: Embedder {
	fn try_new(
		model_repo_name: &str,
	) -> Result<DynEmbedder>;
}

pub struct DynEmbedder(Box<dyn Embedder>);

impl DynLoadEmbedder for DynEmbedder {
	fn try_new(
		model_repo_name: &str,
	) -> Result<DynEmbedder>
	{
		match model_repo_name {
			"jinaai/jina-embeddings-v2-base-en" => {
				Ok(Self(Box::new(JinaBertBaseV2::new()?)))
			},
			"sentence-transformers/all-MiniLM-L6-v2" => {
				Ok(Self(Box::new(AllMiniLmL6V2::new()?)))
			}
			_ => {Err(anyhow!("No matching model found."))}
		}
	}
}

impl Embedder for DynEmbedder {
	fn forward(&self, xs: &Tensor) -> Result<Tensor> {
		self.0.as_ref().forward(xs)
	}
}

pub struct JinaBertBaseV2(_JinaBertModel);

impl StaticLoadEmbedder for JinaBertBaseV2 {
	fn new() -> Result<Self> {
		let model_path = Api::new()?
			.repo(Repo::new("jinaai/jina-embeddings-v2-base-en".into(), RepoType::Model))
			.get("model.safetensors")?;

		let vb =
			unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &DEVICE)? };

		let cfg = JinaBertConfig::v2_base();
		let model = _JinaBertModel::new(vb, &cfg)?;

		Ok(Self(model))
	}
}

impl Embedder for JinaBertBaseV2 {
	fn forward(&self, xs: &Tensor) -> Result<Tensor> {
		Ok(self.0.forward(xs)?)
	}
}

pub struct AllMiniLmL6V2(BertModel);

impl Embedder for AllMiniLmL6V2 {
	fn forward(&self, xs: &Tensor) -> Result<Tensor> {
		let token_type_ids = xs.zeros_like()?;
		Ok(self.0.forward(xs, &token_type_ids)?)
	}
}


impl StaticLoadEmbedder for AllMiniLmL6V2 {
	fn new() -> Result<Self> {
		let default_revision = "refs/pr/21".to_string();
		let api = Api::new()?
			.repo(Repo::with_revision("sentence-transformers/all-MiniLM-L6-v2".into(), RepoType::Model, default_revision));

		let model_path = api
			.get("model.safetensors")?;

		let config_path = api
			.get("config.json")?;

		let config_str = std::fs::read_to_string(config_path)?;
		let cfg: BertConfig = serde_json::from_str(&config_str)?;

		let vb =
			unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &DEVICE)? };

		let model = BertModel::load(vb, &cfg)?;

		Ok(Self(model))
	}
}