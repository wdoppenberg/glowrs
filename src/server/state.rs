use crate::infer::Infer;


#[derive(Clone)]
pub struct ServerState {
	pub infer: Infer,
}

impl Default for ServerState {
	fn default() -> Self {
		let infer = Infer::new();
		Self { infer }
	}
}

