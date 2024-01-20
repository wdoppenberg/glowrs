use anyhow::{Error as E, Result};
use candle_core::backend::BackendDevice;
use candle_core::{Device, DType, Tensor};
use candle_core::metal_backend::MetalDevice;
use candle_nn::VarBuilder;
use hf_hub::{Repo, RepoType};
use hf_hub::api::sync::Api;
use once_cell::sync::Lazy;
use tokenizers::tokenizer::Tokenizer;
use crate::embedding::sbert::SBert;

#[cfg(feature = "metal")]
static DEVICE: Lazy<Device> = Lazy::new(|| Device::Metal(MetalDevice::new(0).expect("No Metal device found.")));

#[cfg(not(any(feature = "metal")))]
static DEVICE: Lazy<Device> = Lazy::new(|| Device::Cpu);

pub struct SentenceTransformer<M>
where M: SBert
{
	model: M,
	tokenizer: Tokenizer,
}

pub struct Args {
	/// L2 normalization for embeddings.
	pub normalize_embeddings: bool,

	/// Tokenizer name
	pub tokenizer: Option<String>,

	/// Jina Embedder model name
	pub model: Option<String>,
}

impl<M> SentenceTransformer<M>
where M: SBert
{
	pub fn try_new() -> Result<Self> {

		let model_path = Api::new()?
				.repo(Repo::new(
					M::model_repo_name(),
					RepoType::Model,
				))
				.get("model.safetensors")?;

		let tokenizer_path = Api::new()?
				.repo(Repo::new(
					M::tokenizer_repo_name(),
					RepoType::Model,
				))
				.get("tokenizer.json")?;

		let config = M::default_config();
		let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

		if let Some(pp) = tokenizer.get_padding_mut() {
			pp.strategy = tokenizers::PaddingStrategy::BatchLongest
		} else {
			let pp = tokenizers::PaddingParams {
				strategy: tokenizers::PaddingStrategy::BatchLongest,
				..Default::default()
			};
			tokenizer.with_padding(Some(pp));
		}

		let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &DEVICE)? };
		let model = M::new(vb, &config)?;

		Ok(Self {
			model,
			tokenizer,
		})
	}
	pub fn encode_batch(&self, sentences: Vec<&str>, normalize: bool) -> Result<Tensor> {
		let tokens = self.tokenizer
			.encode_batch(sentences.to_vec(), true)
			.map_err(E::msg)?;

		let token_ids = tokens
			.iter()
			.map(|tokens| {
				let tokens = tokens.get_ids().to_vec();
				Tensor::new(tokens.as_slice(), &DEVICE)
			})
			.collect::<candle_core::Result<Vec<_>>>()?;

		let token_ids = Tensor::stack(&token_ids, 0)?;
		tracing::trace!("running inference on batch {:?}", token_ids.shape());
		let embeddings = self.model.forward(&token_ids)?;
		tracing::trace!("generated embeddings {:?}", embeddings.shape());

		// Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
		let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
		let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
		let embeddings = if normalize {
			normalize_l2(&embeddings)?
		} else {
			embeddings
		};
		Ok(embeddings)
	}
}

pub fn normalize_l2(v: &Tensor) -> candle_core::Result<Tensor> {
	v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
}

#[cfg(test)]
mod test {
	use super::*;
	use std::time::Instant;
	use crate::embedding::sbert::JinaBertBaseV2;

	#[test]
	fn test_embedder() -> Result<()> {
		let start = Instant::now();

		let embedder: SentenceTransformer<JinaBertBaseV2> = SentenceTransformer::try_new()?;

		let sentences = vec![
			"The cat sits outside",
			"A man is playing guitar",
			"I love pasta",
			"The new movie is awesome",
			"The cat plays in the garden",
			"A woman watches TV",
			"The new movie is so great",
			"Do you like pizza?",
		];

		let model_load_duration = Instant::now() - start;
		dbg!(format!("Model loaded in {}ms", model_load_duration.as_millis()));

		let embeddings = embedder.encode_batch(sentences, true)?;

		dbg!(format!("Pooled embeddings {:?}", embeddings.shape()));
		dbg!(format!("Inference done in {}ms", (Instant::now() - start - model_load_duration).as_millis()));

		Ok(())
	}
}
