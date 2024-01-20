<div align="center">

# 🚧`glowrs`🚧

</div>


> An experimental Rust web server for embedding sentences

An all-Rust web server for sentence embedding inference. Uses
[`candle`](https://github.com/huggingface/candle) as DL framework. Currently runs `jina-embeddings-v2-base-en` but 
it will support other sentence embedders later.

## Features

- [X] OpenAI compatible (`/v1/embeddings`) REST API endpoint
- [X] `candle` inference with Jina AI embeddings
- [X] Hardware acceleration (Metal for now)
- [ ] Queueing
- [ ] Multiple model

## Usage

```bash
cargo run --bin server --release
```

### `curl`
```shell
curl -X POST http://localhost:3000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["The food was delicious and the waiter...", "was too"], 
    "model": "jina-embeddings-v2-base-en",
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
	model="jina-embeddings-v2-base-en"
))

print(f"Done in {time() - start}")
```

## Disclaimer

This is still a work-in-progress. The embedding performance is decent but can probably do with some
benchmarking. Furthermore, for higher batch sizes, the program is killed due to a [bug](https://github.com/huggingface/candle/issues/1596).

Do not use this in a production environment. 