//! # `glowrs`
//! 
//! `glowrs` provides an easy and familiar interface to use pre-trained models for embeddings and sentence similarity.
//! 
//! ## Example
//! 
//! ```rust
//! use glowrs::SentenceTransformer;
//! 
//! let encoder = SentenceTransformer::from_repo_string("sentence-transformers/all-MiniLM-L6-v2").unwrap();
//! 
//! let sentences = vec![
//!    "Hello, how are you?",
//!    "Hey, how are you doing?"
//! ];
//! 
//! let embeddings = encoder.encode_batch(sentences, true).unwrap();
//! 
//! println!("{:?}", embeddings);
//! ```
//! 
//! ## Features
//! 
//! - Load models from Hugging Face Hub
//! - Spin up a REST API server to serve embeddings
//! - More to come!
//! 
//! ## Running the server
 
#![doc = include_str!("../README.md")]

pub mod model;
pub mod server;


pub use model::sentence_transformer::SentenceTransformer;

