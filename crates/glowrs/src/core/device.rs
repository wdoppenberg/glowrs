use candle_core::Device;
use once_cell::sync::Lazy;

#[cfg(all(feature = "metal", feature = "cuda"))]
compile_error!("feature \"metal\" and feature \"cuda\" cannot be enabled at the same time");

#[cfg(feature = "metal")]
pub static DEVICE: Lazy<Device> =
    Lazy::new(|| Device::new_metal(0).expect("No Metal device found."));

#[cfg(feature = "cuda")]
pub static DEVICE: Lazy<Device> = Lazy::new(|| Device::new_cuda(0).expect("No CUDA device found."));

#[cfg(not(any(feature = "metal", feature = "cuda")))]
pub static DEVICE: Lazy<Device> = Lazy::new(|| Device::Cpu);

pub fn print_device_info() {
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    tracing::info!("Using CPU");

    #[cfg(feature = "cuda")]
    tracing::info!("Using CUDA");

    #[cfg(feature = "metal")]
    tracing::info!("Using Metal");
}
