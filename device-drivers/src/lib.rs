//! VitalStream Device Drivers
//! 
//! FDA/IEC 62304 compliant medical device driver library
//! Provides safe, deterministic interfaces for medical device communication

pub mod error;
pub mod traits;
pub mod types;
pub mod protocols;

pub use error::{DeviceError, Result};
pub use traits::{MedicalDevice, DeviceConnection, SignalProcessor};
pub use types::{DeviceInfo, VitalSign, SignalQuality};

/// Library version for regulatory compliance tracking
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the device driver library
/// 
/// This must be called before using any device drivers
pub fn init() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();
    
    tracing::info!("VitalStream Device Drivers v{} initialized", VERSION);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
