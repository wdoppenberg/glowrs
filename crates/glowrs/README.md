# `glowrs`

The `glowrs` library provides an easy and familiar interface to use pre-trained models for embeddings and sentence similarity.
 
## Example

```rust
use glowrs::SentenceTransformer;

fn main() {
    let encoder = SentenceTransformer::from_repo_string("sentence-transformers/all-MiniLM-L6-v2").unwrap();

    let sentences = vec![
        "Hello, how are you?",
        "Hey, how are you doing?"
    ];

    let embeddings = encoder.encode_batch(sentences, true).unwrap();

    println!("{:?}", embeddings);
}
```

## Features
 
- Load models from Hugging Face Hub
- More to come!

### Build features

* `metal`: Compile with Metal acceleration
* `cuda`: Compile with CUDA acceleration
* `accelerate`: Compile with Accelerate framework acceleration (CPU)

## Disclaimer

This is still a work-in-progress. The embedding performance is decent but can probably do with some
benchmarking. Furthermore, for higher batch sizes, the program is killed due to a [bug](https://github.com/huggingface/candle/issues/1596).

Do not use this in a production environment. 