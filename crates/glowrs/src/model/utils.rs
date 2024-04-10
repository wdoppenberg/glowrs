use crate::model::embedder::EmbedderType;
use candle_core::Tensor;

pub fn normalize_l2(v: &Tensor) -> candle_core::Result<Tensor> {
    v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
}

pub fn parse_repo_string(repo_string: &str) -> anyhow::Result<(&str, &str, EmbedderType)> {
    // Fail if the repo string is empty
    if repo_string.is_empty() {
        return Err(anyhow::anyhow!("Model repository string is empty"));
    }

    // Fail if the repo string contains illegal characters
    const ILLEGAL_CHARS: [char; 6] = ['\\', '<', '>', '|', '?', '*'];
    if repo_string.chars().any(|c| ILLEGAL_CHARS.contains(&c)) {
        return Err(anyhow::anyhow!(
            "Model repository string contains illegal characters"
        ));
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
        }
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
    use crate::model::embedder::EmbedderType;
    use crate::model::utils::parse_repo_string;

    #[test]
    fn test_parse_repo_string() -> anyhow::Result<()> {
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
