use candle_core::{Device, DType, Module, Tensor};
use candle_transformers::models::jina_bert::{BertModel, Config};
use tokenizers::Tokenizer;
use anyhow::Error as E;
use anyhow::Result;
use candle_nn::VarBuilder;

pub struct Embedder {
	model: BertModel,
	tokenizer: Tokenizer,
}

pub trait SentenceTransformer
{
	fn encode_batch(&self, sentences: Vec<&str>, normalize: bool) -> Result<Tensor>;
}

impl SentenceTransformer for Embedder
{
	fn encode_batch(&self, sentences: Vec<&str>, normalize: bool) -> Result<Tensor> {
		let device = &self.model.device;

		let tokens = self.tokenizer
			.encode_batch(sentences.to_vec(), true)
			.map_err(E::msg)?;

		let token_ids = tokens
			.iter()
			.map(|tokens| {
				let tokens = tokens.get_ids().to_vec();
				Tensor::new(tokens.as_slice(), device)
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

pub struct Args {
	/// Run on CPU rather than on GPU.
	pub cpu: bool,

	/// Enable tracing (generates a trace-timestamp.json file).
	pub tracing: bool,

	/// L2 normalization for embeddings.
	pub normalize_embeddings: bool,

	/// Tokenizer name
	pub tokenizer: Option<String>,

	/// Jina Embedder model name
	pub model: Option<String>,
}


impl Embedder {
	pub fn try_new(args: Args) -> Result<Self> {
		use hf_hub::{api::sync::Api, Repo, RepoType};
		let model = match &args.model {
			Some(model_file) => std::path::PathBuf::from(model_file),
			None => Api::new()?
				.repo(Repo::new(
					"jinaai/jina-embeddings-v2-base-en".to_string(),
					RepoType::Model,
				))
				.get("model.safetensors")?,
		};
		let tokenizer = match &args.tokenizer {
			Some(file) => std::path::PathBuf::from(file),
			None => Api::new()?
				.repo(Repo::new(
					"sentence-transformers/all-MiniLM-L6-v2".to_string(),
					RepoType::Model,
				))
				.get("tokenizer.json")?,
		};
		let device = Device::Cpu;
		let config = Config::v2_base();
		let mut tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;

		if let Some(pp) = tokenizer.get_padding_mut() {
			pp.strategy = tokenizers::PaddingStrategy::BatchLongest
		} else {
			let pp = tokenizers::PaddingParams {
				strategy: tokenizers::PaddingStrategy::BatchLongest,
				..Default::default()
			};
			tokenizer.with_padding(Some(pp));
		}

		let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? };
		let model = BertModel::new(vb, &config)?;

		Ok(Self {
			model,
			tokenizer,
		})
	}
}

pub fn normalize_l2(v: &Tensor) -> candle_core::Result<Tensor> {
	v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
}

#[cfg(test)]
mod test {
	use std::time::Instant;
	use super::*;
	#[test]
	fn test_embedder() -> Result<()> {
		let start = Instant::now();

		let args = Args {
			cpu: true,
			tracing: true,
			normalize_embeddings: true,
			tokenizer: None,
			model: None,
		};

		let embedder = Embedder::try_new(args)?;

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