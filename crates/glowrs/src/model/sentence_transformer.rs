use anyhow::Result;
use candle_core::Tensor;
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use tokenizers::tokenizer::Tokenizer;

use crate::Usage;
use crate::model::embedder::{EmbedderModel, EmbedderType, encode_batch, encode_batch_with_usage, load_model_and_tokenizer};
use crate::Sentences;

/// The SentenceTransformer struct is the main entry point for using pre-trained models for embeddings and sentence similarity.
/// 
/// ## Example
/// 
/// ```rust
///  # use glowrs::SentenceTransformer;
/// 
///  let encoder = SentenceTransformer::from_repo_string("sentence-transformers/all-MiniLM-L6-v2").unwrap();
/// 
///  let sentences = vec![
///     "Hello, how are you?",
///     "Hey, how are you doing?"
///  ];
/// 
///  let normalize = false;
///  let embeddings = encoder.encode_batch(sentences, normalize).unwrap();
/// 
///  println!("{:?}", embeddings);
///  ```
/// 
pub struct SentenceTransformer
{
	name: String,
	model: Box<dyn EmbedderModel>,
	tokenizer: Tokenizer,
}

impl SentenceTransformer
{
	pub fn new(name: String, model: Box<dyn EmbedderModel>, tokenizer: Tokenizer) -> Self {
		Self {
			name,
			model,
			tokenizer
		}
	}

	/// Load a SentenceTransformer model from the Hugging Face Hub.
	/// 
	/// ## Example
	/// 
	/// ```rust
	/// # use glowrs::SentenceTransformer;
	/// 
	/// # fn main() -> anyhow::Result<()> {
	/// let encoder = SentenceTransformer::from_repo_string("sentence-transformers/all-MiniLM-L6-v2")?;
	/// 
	/// # Ok(())
	/// # }
	/// ```
	/// 
	/// If you want to use a specific revision of the model, you can specify it in the repo string:
	/// 
	/// 
	pub fn from_repo_string(repo_string: &str) -> Result<Self> {
		let (model_repo, default_revision, embedder_type) = parse_repo_string(repo_string)?;
		Self::from_repo(model_repo, default_revision, embedder_type)
	}
	
	pub fn from_repo(repo_name: &str, revision: &str, embedder_type: EmbedderType) -> Result<Self> {
		let api = Api::new()?
			.repo(Repo::with_revision(repo_name.into(), RepoType::Model, revision.into()));

		let (model, tokenizer) = load_model_and_tokenizer(api, embedder_type)?;
		Ok(Self::new(repo_name.into(), model, tokenizer))
	}

	pub fn encode_batch_with_usage(
		&self,
		sentences: impl Into<Sentences>,
		normalize: bool,
	) -> Result<(Tensor, Usage)> {
		let (embeddings, usage) = encode_batch_with_usage(
			self.model.as_ref(),
			&self.tokenizer,
			sentences.into(),
			normalize
		)?;
		Ok((embeddings, usage))
	}

	pub fn encode_batch(&self, sentences: impl Into<Sentences>, normalize: bool) -> Result<Tensor> {
		encode_batch(
			self.model.as_ref(),
			&self.tokenizer,
			sentences.into(),
			normalize
		)
	}
	
	pub fn get_name(&self) -> String {
		self.name.clone()
	}
}


fn parse_repo_string(repo_string: &str) -> Result<(&str, &str, EmbedderType)> {
	// Fail if the repo string is empty
	if repo_string.is_empty() {
		return Err(anyhow::anyhow!("Model repository string is empty"));
	}
	
	// Fail if the repo string contains illegal characters
	const ILLEGAL_CHARS: [char; 6] = ['\\', '<', '>', '|', '?', '*'];
	if repo_string.chars().any(|c| ILLEGAL_CHARS.contains(&c)) {
		return Err(anyhow::anyhow!("Model repository string contains illegal characters"));
	}
	
	// Split the repo string by colon
	let parts: Vec<&str> = repo_string.split(':').collect();
	let model_repo = parts[0];
	let mut revision = *parts.get(1).unwrap_or(&"main");
	
	// If revision is an empty string, set it to "main"
	if revision.is_empty() {
		revision = "main";
	}
	
	let embedder_type_str = parts.get(2).cloned();
	
	let embedder_type = match embedder_type_str {
	    None => {
		    // If the model repo contains "jinaai", use JinaBert, otherwise use Bert
	        if model_repo.contains("jinaai") {
	             EmbedderType::JinaBert
	        } else {
	            EmbedderType::Bert
	        }
	    },
	    Some(embedder_type) => {
		    // Match the embedder type string to the EmbedderType enum
	        match &*embedder_type.to_lowercase() {
	            "bert" => EmbedderType::Bert,
	            "jinabert" => EmbedderType::JinaBert,
	            _ => return Err(anyhow::anyhow!("Invalid embedder type")),
	        }
	    }
	};
	
	Ok((model_repo, revision, embedder_type))
}

#[cfg(test)]
mod test {
	use super::*;
	use std::time::Instant;

	#[test]
	fn test_sentence_transformer() -> Result<()> {
		let start = Instant::now();

		let model_repo = "sentence-transformers/all-MiniLM-L6-v2";
		let default_revision = "main";
		let sentence_transformer: SentenceTransformer = SentenceTransformer::from_repo(
			model_repo, 
			default_revision,
			EmbedderType::Bert
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
		println!(
			"Model loaded in {}ms",
			model_load_duration.as_millis()
		);

		let embeddings = sentence_transformer.encode_batch(sentences, true)?;

		println!("Pooled embeddings {:?}", embeddings.shape());
		println!(
			"Inference done in {}ms",
			(Instant::now() - start - model_load_duration).as_millis()
		);

		Ok(())
	}
	
	#[test]
	fn test_parse_repo_string() -> Result<()> {
		let repo_string = "sentence-transformers/all-MiniLM-L6-v2:refs/pr/21";
		let (model_repo, default_revision, et) = parse_repo_string(repo_string)?;
		assert_eq!(model_repo, "sentence-transformers/all-MiniLM-L6-v2");
		assert_eq!(default_revision, "refs/pr/21");
		assert_eq!(et, EmbedderType::Bert);
		
		let repo_string = "sentence-transformers/all-MiniLM-L6-v2";
		let (model_repo, default_revision, et) = parse_repo_string(repo_string)?;
		assert_eq!(model_repo, "sentence-transformers/all-MiniLM-L6-v2");
		assert_eq!(default_revision, "main");
		assert_eq!(et, EmbedderType::Bert);
		
		let repo_string = "sentence-transformers/all-MiniLM-L6-v2:";
		let (model_repo, default_revision, et) = parse_repo_string(repo_string)?;
		assert_eq!(model_repo, "sentence-transformers/all-MiniLM-L6-v2");
		assert_eq!(default_revision, "main");
		assert_eq!(et, EmbedderType::Bert);
		
		let repo_string = "sentence-transformers/all-MiniLM-L6-v2::jinabert";
		let (model_repo, default_revision, et) = parse_repo_string(repo_string)?;
		assert_eq!(model_repo, "sentence-transformers/all-MiniLM-L6-v2");
		assert_eq!(default_revision, "main");
		assert_eq!(et, EmbedderType::JinaBert);
		
		let repo_string = "jinaai/jina-embeddings-v2-base-en";
		let (model_repo, default_revision, et) = parse_repo_string(repo_string)?;
		assert_eq!(model_repo, "jinaai/jina-embeddings-v2-base-en");
		assert_eq!(default_revision, "main");
		assert_eq!(et, EmbedderType::JinaBert);
		
		let repo_string = "jinaai/jina-embeddings-v2-base-en:refs/pr/21:bert";
		let (model_repo, default_revision, et) = parse_repo_string(repo_string)?;
		assert_eq!(model_repo, "jinaai/jina-embeddings-v2-base-en");
		assert_eq!(default_revision, "refs/pr/21");
		assert_eq!(et, EmbedderType::Bert);
		
		let repo_string = "jinaai/jina-embeddings-v2-base-en:refs/pr/21:Bert";
		let (model_repo, default_revision, et) = parse_repo_string(repo_string)?;
		assert_eq!(model_repo, "jinaai/jina-embeddings-v2-base-en");
		assert_eq!(default_revision, "refs/pr/21");
		assert_eq!(et, EmbedderType::Bert);
		
		let illegal_repo_string = "jinaai/jina-embeddings-v2-base-en:refs/pr/21:Bert*";
		assert!(parse_repo_string(illegal_repo_string).is_err());
		
		Ok(())
	}
}


