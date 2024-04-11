use candle_core::Tensor;

pub fn normalize_l1(v: &Tensor) -> candle_core::Result<Tensor> {
    v.broadcast_div(&v.abs()?.sum_keepdim(1)?)
}

pub fn normalize_l2(v: &Tensor) -> candle_core::Result<Tensor> {
    v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
}

pub fn parse_repo_string(repo_string: &str) -> anyhow::Result<(&str, &str)> {
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

    Ok((model_repo, revision))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parse_repo_string() -> anyhow::Result<()> {
        let repo_string = "sentence-transformers/all-MiniLM-L6-v2:refs/pr/21";
        let (model_repo, default_revision) = parse_repo_string(repo_string)?;
        assert_eq!(model_repo, "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(default_revision, "refs/pr/21");

        let repo_string = "sentence-transformers/all-MiniLM-L6-v2";
        let (model_repo, default_revision) = parse_repo_string(repo_string)?;
        assert_eq!(model_repo, "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(default_revision, "main");

        let repo_string = "sentence-transformers/all-MiniLM-L6-v2:";
        let (model_repo, default_revision) = parse_repo_string(repo_string)?;
        assert_eq!(model_repo, "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(default_revision, "main");

        let repo_string = "jinaai/jina-embeddings-v2-base-en";
        let (model_repo, default_revision) = parse_repo_string(repo_string)?;
        assert_eq!(model_repo, "jinaai/jina-embeddings-v2-base-en");
        assert_eq!(default_revision, "main");

        let illegal_repo_string = "jinaai/jina-embeddings-v2-base-en:refs/pr/21*";
        assert!(parse_repo_string(illegal_repo_string).is_err());

        Ok(())
    }
}
