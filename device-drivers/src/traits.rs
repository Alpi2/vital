//! Core traits for medical device interfaces
//! 
//! Defines standard interfaces for FDA/IEC 62304 compliant device communication

use async_trait::async_trait;
use crate::{DeviceError, Result};
use crate::types::{DeviceInfo, VitalSign, SignalQuality};

/// Core trait for all medical devices
#[async_trait]
pub trait MedicalDevice: Send + Sync {
    /// Get device information
    async fn get_info(&self) -> Result<DeviceInfo>;
    
    /// Connect to the device
    async fn connect(&mut self) -> Result<()>;
    
    /// Disconnect from the device
    async fn disconnect(&mut self) -> Result<()>;
    
    /// Check if device is connected
    fn is_connected(&self) -> bool;
    
    /// Start data acquisition
    async fn start_acquisition(&mut self) -> Result<()>;
    
    /// Stop data acquisition
    async fn stop_acquisition(&mut self) -> Result<()>;
    
    /// Read vital signs data
    async fn read_data(&mut self) -> Result<Vec<VitalSign>>;
    
    /// Perform device self-test
    async fn self_test(&mut self) -> Result<bool>;
    
    /// Calibrate device
    async fn calibrate(&mut self) -> Result<()>;
}

/// Trait for device connection management
#[async_trait]
pub trait DeviceConnection: Send + Sync {
    /// Discover available devices
    async fn discover(&self) -> Result<Vec<DeviceInfo>>;
    
    /// Connect to a specific device
    async fn connect(&mut self, device_id: &str) -> Result<()>;
    
    /// Disconnect from device
    async fn disconnect(&mut self) -> Result<()>;
    
    /// Send raw data to device
    async fn send(&mut self, data: &[u8]) -> Result<usize>;
    
    /// Receive raw data from device
    async fn receive(&mut self, buffer: &mut [u8]) -> Result<usize>;
    
    /// Get connection status
    fn is_connected(&self) -> bool;
}

/// Trait for signal processing and quality assessment
pub trait SignalProcessor: Send + Sync {
    /// Calculate Signal Quality Index (SQI)
    fn calculate_sqi(&self, signal: &[f32]) -> Result<SignalQuality>;
    
    /// Detect artifacts in signal
    fn detect_artifacts(&self, signal: &[f32]) -> Result<Vec<usize>>;
    
    /// Remove baseline wander
    fn remove_baseline_wander(&self, signal: &[f32]) -> Result<Vec<f32>>;
    
    /// Filter powerline interference (50/60 Hz)
    fn filter_powerline(&self, signal: &[f32], frequency: f32) -> Result<Vec<f32>>;
    
    /// Remove motion artifacts
    fn remove_motion_artifacts(&self, signal: &[f32]) -> Result<Vec<f32>>;
    
    /// Detect lead-off condition
    fn detect_lead_off(&self, signal: &[f32]) -> Result<bool>;
    
    /// Detect pacemaker spikes
    fn detect_pacemaker_spikes(&self, signal: &[f32]) -> Result<Vec<usize>>;
}

/// Trait for ECG-specific devices
#[async_trait]
pub trait ECGDevice: MedicalDevice {
    /// Get number of ECG leads
    fn get_lead_count(&self) -> u8;
    
    /// Get sampling rate in Hz
    fn get_sampling_rate(&self) -> u32;
    
    /// Read ECG waveform data
    async fn read_ecg_data(&mut self) -> Result<Vec<Vec<f32>>>;
    
    /// Set lead configuration (3-lead, 5-lead, 12-lead)
    async fn set_lead_config(&mut self, leads: u8) -> Result<()>;
}

/// Trait for multi-parameter monitors
#[async_trait]
pub trait MultiParameterMonitor: MedicalDevice {
    /// Get supported vital sign types
    fn get_supported_parameters(&self) -> Vec<String>;
    
    /// Enable/disable specific parameter monitoring
    async fn set_parameter_enabled(&mut self, parameter: &str, enabled: bool) -> Result<()>;
    
    /// Get all current vital signs
    async fn get_all_vitals(&mut self) -> Result<Vec<VitalSign>>;
}
