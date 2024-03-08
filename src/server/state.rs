use crate::infer::Infer;
use anyhow::Result;


/// Represents the state of the server.
#[derive(Clone)]
pub struct ServerState {
	pub infer: Infer,
}


impl ServerState {
	pub fn new() -> Result<Self> {
		let infer = Infer::new()?;
		
		Ok(Self { infer })
	}
}