use crate::model::embedder::{EmbedderModel, encode_batch, encode_batch_with_usage, load_model_and_tokenizer};
use anyhow::Result;
use candle_core::Tensor;
use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::{Repo, RepoType};
use tokenizers::tokenizer::Tokenizer;
use crate::server::data_models::{Sentences, Usage};


pub struct SentenceTransformer<E>
where E: EmbedderModel
{
	model: E,
	tokenizer: Tokenizer,
}

impl<E> SentenceTransformer<E>
	where
		E: EmbedderModel,
{
	pub fn new(model: E, tokenizer: Tokenizer) -> Self {
		Self {
			model,
			tokenizer
		}
	}

	pub fn from_repo(repo_name: impl Into<String>, revision: impl Into<String>) -> Result<Self> {
		let api = Api::new()?
			.repo(Repo::with_revision(repo_name.into(), RepoType::Model, revision.into()));

		Self::try_from(api)
	}

	pub fn encode_batch_with_usage(
		&self,
		sentences: Sentences,
		normalize: bool,
	) -> Result<(Tensor, Usage)> {
		let (embeddings, usage) = encode_batch_with_usage(
			&self.model,
			&self.tokenizer,
			sentences,
			normalize
		)?;
		Ok((embeddings, usage))
	}

	pub fn encode_batch(&self, sentences: Sentences, normalize: bool) -> Result<Tensor> {
		encode_batch(
			&self.model,
			&self.tokenizer,
			sentences,
			normalize
		)
	}
}

impl<E> TryFrom<ApiRepo> for SentenceTransformer<E>
	where
		E: EmbedderModel,
{
	type Error = anyhow::Error;
	fn try_from(api: ApiRepo) -> Result<Self> {
		let (model, tokenizer) = load_model_and_tokenizer(api)?;
		Ok(Self::new(model, tokenizer))
	}
}

#[cfg(test)]
mod test {
	use super::*;
	use std::time::Instant;
	use candle_transformers::models::bert::BertModel;

	#[test]
	fn test_sentence_transformer() -> Result<()> {
		let start = Instant::now();

		let model_repo = "sentence-transformers/all-MiniLM-L6-v2";
		let default_revision = "refs/pr/21".to_string();
		let sentence_transformer: SentenceTransformer<BertModel> = SentenceTransformer::from_repo(
			model_repo, default_revision
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


