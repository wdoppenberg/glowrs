<div align="center">

# 🚧`glowrs`🚧

</div>


> An experimental Rust web server for embedding sentences

An all-Rust web server for sentence embedding inference. Uses
[`candle`](https://github.com/huggingface/candle) as DL framework.

## Features

- [X] OpenAI compatible (`/v1/embeddings`) REST API endpoint
- [X] `candle` inference with Jina AI embeddings
- [ ] Hardware acceleration
- [ ] Queueing
- [ ] Multiple model

## Usage

```bash
cargo run --bin server --release
```

### `curl`
```shell

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
	input=["This is a sentence that requires an embedding"] * 40,
	model="<placeholder>"
))

print(f"Done in {time() - start}")
```

## Disclaimer

This is still a work-in-progress. The embedding performance is not great and does not scale well. This is ofcourse partially
due to poorly optimized code, but also due to the DL backend. In the future, perhaps upstream
optimizations to the `candle` library will improve this. 

Do not use this in a production environment. 