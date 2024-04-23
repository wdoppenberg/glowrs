use glowrs::{Device, Error, PoolingStrategy, SentenceTransformer};
use std::process::ExitCode;

fn main() -> Result<ExitCode, Error> {
    let sentences = vec![
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
        "The new movie is awesome",
        "The cat plays in the garden",
        "A woman watches TV",
        "The new movie is so great",
        "Do you like pizza?",
        "The cat sits",
    ];
    let device = Device::Cpu;
    let encoder =
        SentenceTransformer::from_repo_string("Snowflake/snowflake-arctic-embed-xs", &device)?;

    let pooling_strategy = PoolingStrategy::Mean;
    let embeddings = encoder.encode_batch(sentences.clone(), false, pooling_strategy)?;
    println!("Embeddings: {:?}", embeddings);

    let (n_sentences, _) = embeddings.dims2()?;
    let mut similarities = Vec::with_capacity(n_sentences * (n_sentences - 1) / 2);

    for i in 0..n_sentences {
        let e_i = embeddings.get(i)?;
        for j in (i + 1)..n_sentences {
            let e_j = embeddings.get(j)?;
            let sum_ij = (&e_i * &e_j)?.sum_all()?.to_scalar::<f32>()?;
            let sum_i2 = (&e_i * &e_i)?.sum_all()?.to_scalar::<f32>()?;
            let sum_j2 = (&e_j * &e_j)?.sum_all()?.to_scalar::<f32>()?;
            let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
            similarities.push((cosine_similarity, i, j))
        }
    }

    similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
    for &(score, i, j) in similarities[..5].iter() {
        println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
    }

    Ok(ExitCode::SUCCESS)
}
