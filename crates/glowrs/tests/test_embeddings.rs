use candle_core::Tensor;
use serde::Deserialize;
use std::process::ExitCode;

use glowrs::model::utils::normalize_l2;
use glowrs::{PoolingStrategy, Result};

#[derive(Deserialize)]
struct EmbeddingsExample {
    sentence: String,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct EmbeddingsFixture {
    model: String,
    examples: Vec<EmbeddingsExample>,
}

#[derive(Deserialize)]
struct Examples {
    fixtures: Vec<EmbeddingsFixture>,
}

#[test]
fn test_similarity_sentence_transformers() -> Result<ExitCode> {
    use approx::assert_relative_eq;
    let examples: Examples =
        serde_json::from_str(include_str!("./fixtures/embeddings/examples.json"))?;
    let device = glowrs::Device::Cpu;
    for fixture in examples.fixtures {
        let encoder = glowrs::SentenceTransformer::from_repo_string(&fixture.model, &device)?;
        println!("Loaded model: {}", &fixture.model);
        for example in fixture.examples {
            let embedding =
                encoder.encode_batch(vec![example.sentence], false, PoolingStrategy::Mean)?;
            let embedding = normalize_l2(&embedding)?;

            let expected_dim = example.embedding.len();
            let expected = Tensor::from_vec(example.embedding, (1, expected_dim), &device)?;
            let expected = normalize_l2(&expected)?;

            assert_eq!(embedding.dims(), expected.dims());

            let sim = embedding.matmul(&expected.t()?)?.squeeze(1)?;

            let sim = sim.to_vec1::<f32>()?;
            let sim = sim.first().expect("Expected a value");
            println!("Similarity: {}", sim);
            assert_relative_eq!(*sim, 1.0, epsilon = 1e-3);
        }
        println!("Passed all examples for model: {}", &fixture.model)
    }

    Ok(ExitCode::SUCCESS)
}
