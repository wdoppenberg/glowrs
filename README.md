<div align="center">

# 🚧`GLOWRS`🚧

</div>


> A work-in-progress Rust library for ML inference.

An all-Rust web server for ML inference. Can use either [`burn`](https://github.com/burn-rs/burn) or 
[`candle`](https://github.com/huggingface/candle) as the backend out of the box. 

## Features

- [ ] Worker pool
- [ ] Task queue
- [ ] `burn` inference templates
- [ ] `candle` inference templates
- [ ] HTTP server using `axum`
- [ ] gRPC server using `tonic`

## Usage

```bash
cargo add glowrs
```

```rust
use glowrs::{Queue, Task};

todo!("Write a README");
```

## Disclaimer

This is still a work-in-progress. The API is not stable and will change.