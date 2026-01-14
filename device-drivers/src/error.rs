//! Error types for device driver operations
//! 
//! Comprehensive error handling for FDA/IEC 62304 compliance

use thiserror::Error;

/// Result type alias for device operations
pub type Result<T> = std::result::Result<T, DeviceError>;

/// Comprehensive error types for medical device operations
#[derive(Error, Debug)]
pub enum DeviceError {
    /// Device connection errors
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Device not found: {0}")]
    DeviceNotFound(String),
    
    #[error("Device disconnected: {0}")]
    DeviceDisconnected(String),
    
    /// Protocol errors
    #[error("Protocol error: {0}")]
    ProtocolError(String),
    
    #[error("Invalid data format: {0}")]
    InvalidDataFormat(String),
    
    #[error("Checksum mismatch")]
    ChecksumMismatch,
    
    /// Signal quality errors
    #[error("Signal quality too low: {0}")]
    LowSignalQuality(f32),
    
    #[error("Lead off detected on channel {0}")]
    LeadOff(u8),
    
    #[error("Artifact detected: {0}")]
    ArtifactDetected(String),
    
    /// Hardware errors
    #[error("Hardware error: {0}")]
    HardwareError(String),
    
    #[error("Calibration required")]
    CalibrationRequired,
    
    /// Timeout errors
    #[error("Operation timeout after {0}ms")]
    Timeout(u64),
    
    /// Configuration errors
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    
    /// Safety-critical errors (FDA/IEC 62304 Class C)
    #[error("CRITICAL: Safety violation - {0}")]
    SafetyViolation(String),
    
    /// IO errors
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Serialization errors
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    /// Generic errors
    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl DeviceError {
    /// Check if error is safety-critical
    pub fn is_critical(&self) -> bool {
        matches!(self, DeviceError::SafetyViolation(_))
    }
    
    /// Get error severity level (1-5, 5 being most critical)
    pub fn severity(&self) -> u8 {
        match self {
            DeviceError::SafetyViolation(_) => 5,
            DeviceError::HardwareError(_) => 4,
            DeviceError::LeadOff(_) => 4,
            DeviceError::DeviceDisconnected(_) => 3,
            DeviceError::LowSignalQuality(_) => 3,
            DeviceError::ProtocolError(_) => 2,
            _ => 1,
        }
    }
}
