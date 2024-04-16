# `glowrs`

# Library Usage

 
`glowrs` provides an easy and familiar interface to use pre-trained models for embeddings and sentence similarity.
 
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
benchmarking. Furthermore, for higher batch sizes, the program is killed due to a [bug](https://github.com/huggingface/candle/issues/1596).

Do not use this in a production environment. 