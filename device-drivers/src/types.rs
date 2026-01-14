//! Common types for medical device operations

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Unique device identifier
    pub device_id: String,
    
    /// Manufacturer name
    pub manufacturer: String,
    
    /// Device model
    pub model: String,
    
    /// Serial number
    pub serial_number: String,
    
    /// Firmware version
    pub firmware_version: String,
    
    /// Hardware version
    pub hardware_version: String,
    
    /// Device type (ECG, SpO2, NIBP, etc.)
    pub device_type: DeviceType,
    
    /// Connection type
    pub connection_type: ConnectionType,
    
    /// FDA/CE certification info
    pub certifications: Vec<String>,
}

/// Device type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeviceType {
    ECG12Lead,
    ECG3Lead,
    ECG5Lead,
    SpO2,
    NIBP,  // Non-invasive blood pressure
    IBP,   // Invasive blood pressure
    Temperature,
    Respiration,
    Capnography,
    MultiParameter,
    Unknown,
}

/// Connection type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConnectionType {
    BLE,
    USB,
    Serial,
    WiFi,
    Ethernet,
}

/// Vital sign measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitalSign {
    /// Type of vital sign
    pub sign_type: VitalSignType,
    
    /// Measured value
    pub value: f32,
    
    /// Unit of measurement
    pub unit: String,
    
    /// Timestamp of measurement
    pub timestamp: DateTime<Utc>,
    
    /// Signal quality (0.0 - 1.0)
    pub quality: f32,
    
    /// Patient ID (if available)
    pub patient_id: Option<String>,
    
    /// Additional metadata
    pub metadata: Option<serde_json::Value>,
}

/// Vital sign types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VitalSignType {
    HeartRate,
    RespirationRate,
    SpO2,
    Temperature,
    SystolicBP,
    DiastolicBP,
    MeanBP,
    EtCO2,  // End-tidal CO2
    ECGWaveform,
    Custom(String),
}

/// Signal quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalQuality {
    /// Overall quality index (0.0 - 1.0)
    pub sqi: f32,
    
    /// Signal-to-noise ratio (dB)
    pub snr: f32,
    
    /// Artifact percentage
    pub artifact_percentage: f32,
    
    /// Lead-off status
    pub lead_off: bool,
    
    /// Quality assessment timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Detailed quality metrics
    pub metrics: QualityMetrics,
}

/// Detailed quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Baseline wander level
    pub baseline_wander: f32,
    
    /// Powerline interference level
    pub powerline_interference: f32,
    
    /// Motion artifact level
    pub motion_artifact: f32,
    
    /// Muscle noise level
    pub muscle_noise: f32,
    
    /// Electrode contact quality
    pub electrode_contact: f32,
}

/// ECG waveform data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ECGWaveform {
    /// Lead number (I, II, III, aVR, aVL, aVF, V1-V6)
    pub lead: String,
    
    /// Sampling rate in Hz
    pub sampling_rate: u32,
    
    /// Waveform samples (in mV)
    pub samples: Vec<f32>,
    
    /// Start timestamp
    pub start_time: DateTime<Utc>,
    
    /// Signal quality
    pub quality: SignalQuality,
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Sampling rate in Hz
    pub sampling_rate: u32,
    
    /// Number of leads/channels
    pub channel_count: u8,
    
    /// Gain settings
    pub gain: f32,
    
    /// Filter settings
    pub filters: FilterConfig,
    
    /// Alarm thresholds
    pub alarm_thresholds: Option<AlarmThresholds>,
}

/// Filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    /// High-pass filter cutoff (Hz)
    pub highpass_cutoff: f32,
    
    /// Low-pass filter cutoff (Hz)
    pub lowpass_cutoff: f32,
    
    /// Notch filter frequency (50 or 60 Hz)
    pub notch_frequency: Option<f32>,
    
    /// Enable baseline wander removal
    pub baseline_removal: bool,
}

/// Alarm thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlarmThresholds {
    /// Heart rate thresholds (bpm)
    pub hr_low: Option<u32>,
    pub hr_high: Option<u32>,
    
    /// SpO2 thresholds (%)
    pub spo2_low: Option<u32>,
    
    /// Respiration rate thresholds (breaths/min)
    pub rr_low: Option<u32>,
    pub rr_high: Option<u32>,
}
