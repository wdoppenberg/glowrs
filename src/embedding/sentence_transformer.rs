use crate::embedding::embedder::{DynEmbedder, DynLoadEmbedder, Embedder};
use anyhow::{anyhow, Error, Result};
use candle_core::{Tensor};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use tokenizers::tokenizer::Tokenizer;
use crate::routes::{Sentences, Usage};


pub struct SentenceTransformer<E>
where E: Embedder
{
	embedder: E,
	tokenizer: Tokenizer,
}

impl<E> SentenceTransformer<E>
	where
		E: Embedder,
{
	pub fn new(embedder: E, tokenizer: Tokenizer) -> Self {
		Self {
			embedder,
			tokenizer
		}
	}

	// TODO: Fix token usage - currently incorrectly calculated
	pub fn encode_batch_with_usage(
		&self,
		sentences: Sentences,
		normalize: bool,
	) -> Result<(Tensor, Usage)> {
		let (embeddings, usage) = self.embedder.encode_batch_with_usage(
			sentences,
			normalize,
			&self.tokenizer
		)?;
		Ok((embeddings, usage))
	}

	pub fn encode_batch(&self, sentences: Sentences, normalize: bool) -> Result<Tensor> {
		Ok(self.encode_batch_with_usage(sentences, normalize)?.0)
	}
}

#[derive(Default)]
pub struct SentenceTransformerBuilder
{
	embedder: Option<DynEmbedder>,
	tokenizer: Option<Tokenizer>,
}

impl SentenceTransformerBuilder
{
	pub fn new() -> Self {
		Default::default()
	}
	pub fn with_tokenizer_repo(self, repo_name: &str) -> Result<Self> {
		let tokenizer_path = Api::new()?
			.repo(Repo::new(
				repo_name.to_string(),
				RepoType::Model,
			))
			.get("tokenizer.json")?;

		let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(Error::msg)?;

		if let Some(pp) = tokenizer.get_padding_mut() {
			pp.strategy = tokenizers::PaddingStrategy::BatchLongest
		} else {
			let pp = tokenizers::PaddingParams {
				strategy: tokenizers::PaddingStrategy::BatchLongest,
				..Default::default()
			};
			tokenizer.with_padding(Some(pp));
		}
		Ok(Self {
			embedder: self.embedder,
			tokenizer: Some(tokenizer)
		})
	}

	pub fn with_model_repo(self, repo_name: &str) -> Result<Self> {
		let embedder = DynEmbedder::try_new(repo_name)?;

		Ok(Self{
			embedder: Some(embedder),
			tokenizer: self.tokenizer
		})
	}

	pub fn build(self) -> Result<SentenceTransformer<DynEmbedder>> {
		match (self.embedder, self.tokenizer) {
			(Some(embedder), Some(tokenizer)) => {
					Ok(SentenceTransformer {
					embedder,
					tokenizer
				})
			}
			(None, Some(_)) => Err(anyhow!("No embedder repository provided!")),
			(Some(_), None) => Err(anyhow!("No tokenizer repository provided!")),
			_ => Err(anyhow!("Neither embedder or tokenizer repository provided!"))
		}

	}
}


#[cfg(test)]
mod test {
	use super::*;
	use std::time::Instant;

	#[test]
	fn test_sentence_transformer() -> Result<()> {
		let start = Instant::now();

		let builder = SentenceTransformerBuilder::new();
		let sentence_transformer = builder
			.with_model_repo("jinaai/jina-embeddings-v2-base-en")?
			.with_tokenizer_repo("sentence-transformers/all-MiniLM-L6-v2")?
			.build()?;


		let sentences =  Sentences::from(vec![
			"The cat sits outside",
			"A man is playing guitar",
			"I love pasta",
			"The new movie is awesome",
			"The cat plays in the garden",
			"A woman watches TV",
			"The new movie is so great",
			"Do you like pizza?",
		]);

		let model_load_duration = Instant::now() - start;
		dbg!(format!(
			"Model loaded in {}ms",
			model_load_duration.as_millis()
		));

		let embeddings = sentence_transformer.encode_batch(sentences, true)?;

		dbg!(format!("Pooled embeddings {:?}", embeddings.shape()));
		dbg!(format!(
			"Inference done in {}ms",
			(Instant::now() - start - model_load_duration).as_millis()
		));

		Ok(())
	}
}


