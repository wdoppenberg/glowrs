# `glowrs`

# Library Usage

 
`glowrs` provides an easy and familiar interface to use pre-trained models for embeddings and sentence similarity. 
Inspired by the [`sentence-transformers`](https://www.sbert.net/index.html) library, which is a great 
Python library for sentence embeddings and features a wide range of models and utilities. 
 
## Example

```rust
use glowrs::{SentenceTransformer, Device, PoolingStrategy, Error};

fn main() -> Result<(), Error> {
    let encoder = SentenceTransformer::from_repo_string("sentence-transformers/all-MiniLM-L6-v2", &Device::Cpu)?;

    let sentences = vec![
        "Hello, how are you?",
        "Hey, how are you doing?"
    ];

    let embeddings = encoder.encode_batch(sentences, true)?;

    println!("{:?}", embeddings);
    
    Ok(())
}
```


## Features
 
- Load models from Hugging Face Hub
- Use hardware acceleration (Metal, CUDA)
- More to come!

# Server Usage

`glowrs-server`  provides a web server for sentence embedding inference. Uses
[`candle`](https://github.com/huggingface/candle) as Tensor framework. It currently supports Bert type models hosted on Huggingface, such as those provided by 
[`sentence-transformers`](https://huggingface.co/sentence-transformers), 
[`Tom Aarsen`](https://huggingface.co/tomaarsen), or [`Jina AI`](https://huggingface.co/jinaai), as long as they provide safetensors model weights.


Example usage with the `jina-embeddings-v2-base-en` model:

```bash
cargo run --bin glowrs-server --release -- --model-repo jinaai/jina-embeddings-v2-base-en
```

If you want to use a certain revision of the model, you can append it to the repository name like so.

```bash
cargo run --bin glowrs-server --release -- --model-repo jinaai/jina-embeddings-v2-base-en:main
```

If you want to run multiple models, you can run multiple instances of the glowrs-server with different model repos.

```bash
cargo run --bin glowrs-server --release -- --model-repo jinaai/jina-embeddings-v2-base-en sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

**Warning:** This is not supported with `metal` acceleration for now. 

### Instructions:

```shell
Usage: glowrs-server [OPTIONS]

Options:
  -m, --model-repo <MODEL_REPO>  
  -r, --revision <REVISION>      [default: main]
  -h, --help                     Print help
```

### Build features

* `metal`: Compile with Metal acceleration
* `cuda`: Compile with CUDA acceleration
* `accelerate`: Compile with Accelerate framework acceleration (CPU)

## Docker Usage

For now the docker image only supports CPU on x86 and arm64. 

```shell
docker run -p 3000:3000 ghcr.io/wdoppenberg/glowrs-server:latest --model-repo <MODEL_REPO>
```


## Features

- [X] OpenAI API compatible (`/v1/embeddings`) REST API endpoint
- [X] `candle` inference for bert and jina-bert models
- [X] Hardware acceleration (Metal for now)
- [X] Queueing
- [ ] Multiple models
- [ ] Batching
- [ ] Performance metrics


### `curl`
```shell
curl -X POST http://localhost:3000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["The food was delicious and the waiter...", "was too"], 
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "encoding_format": "float"
  }'
```


### Python `openai` client

Install the OpenAI Python library:
```shell
pip install openai
```

Use the embeddings method regularly.
```python
from openai import OpenAI
from time import time

client = OpenAI(
	api_key="sk-something",
	base_url="http://127.0.0.1:3000/v1"
)

start = time()
print(client.embeddings.create(
	input=["This is a sentence that requires an embedding"] * 50,
	model="jinaai/jina-embeddings-v2-base-en"
))

print(f"Done in {time() - start}")

# List models
print(client.models.list())
```

## Details

* Use `TOKIO_WORKER_THREADS` to set the number of threads _per queue_.

## Disclaimer

This is still a work-in-progress. The embedding performance is decent but can probably do with some
benchmarking. Furthermore, this is meant to be a lightweight embedding model library + server. 

Do not use this in a production environment. If you are looking for something production-ready & in Rust, 
consider [`text-embeddings-inference`](https://github.com/huggingface/text-embeddings-inference).

## Credits

* [Huggingface](https://huggingface.co) for the model hosting, the `candle` library, [`text-embeddings-inference`](https://github.com/huggingface/text-embeddings-inference), and 
[`text-generation-inference`](https://github.com/huggingface/text-generation-inference) which has helped me find the right patterns.
* [`sentence-transformers`](https://www.sbert.net/index.html) and its contributors for being the gold standard in sentence embeddings.