<div align="center">

# 🚧`glowrs`🚧

</div>


> An experimental Rust web server for embedding sentences

An all-Rust web server for sentence embedding inference. Uses
[`candle`](https://github.com/huggingface/candle) as the runtime.

## Features

- [X] OpenAI compatible (`/v1/embeddings`) endpoint
- [X] `candle` inference with Jina AI embeddings
- [X] HTTP server using `axum`
- [ ] Hardware acceleration

## Usage

```bash
cargo run --bin server --release
```

## Disclaimer

This is still a work-in-progress. The embedding performance is not great and does not scale well. This is ofcourse partially
due to poorly optimized code, but also due to the DL backend. In the future, perhaps upstream
optimizations to the `candle` library will improve this. 

Do not use this in a production environment. 