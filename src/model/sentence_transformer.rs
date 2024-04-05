use crate::model::embedder::{EmbedderModel, EmbedderType, encode_batch, encode_batch_with_usage, load_model_and_tokenizer};
use anyhow::Result;
use candle_core::Tensor;
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use tokenizers::tokenizer::Tokenizer;

use crate::server::data_models::{Sentences, Usage};


pub struct SentenceTransformer
{
	model: Box<dyn EmbedderModel>,
	tokenizer: Tokenizer,
}

impl SentenceTransformer
{
	pub fn new(model: Box<dyn EmbedderModel>, tokenizer: Tokenizer) -> Self {
		Self {
			model,
			tokenizer
		}
	}
	pub fn from_repo(repo_name: impl Into<String>, revision: impl Into<String>, embedder_type: EmbedderType) -> Result<Self> {
		let api = Api::new()?
			.repo(Repo::with_revision(repo_name.into(), RepoType::Model, revision.into()));

		let (model, tokenizer) = load_model_and_tokenizer(api, embedder_type)?;
		Ok(Self::new(model, tokenizer))
	}

	pub fn encode_batch_with_usage(
		&self,
		sentences: Sentences,
		normalize: bool,
	) -> Result<(Tensor, Usage)> {
		let (embeddings, usage) = encode_batch_with_usage(
			self.model.as_ref(),
			&self.tokenizer,
			sentences,
			normalize
		)?;
		Ok((embeddings, usage))
	}

	pub fn encode_batch(&self, sentences: Sentences, normalize: bool) -> Result<Tensor> {
		encode_batch(
			self.model.as_ref(),
			&self.tokenizer,
			sentences,
			normalize
		)
	}
}

#[cfg(test)]
mod test {
	use super::*;
	use std::time::Instant;

	#[test]
	fn test_sentence_transformer() -> Result<()> {
		let start = Instant::now();

		let model_repo = "sentence-transformers/all-MiniLM-L6-v2";
		let default_revision = "refs/pr/21".to_string();
		let sentence_transformer: SentenceTransformer = SentenceTransformer::from_repo(
			model_repo, default_revision, EmbedderType::Bert
		)?;

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


