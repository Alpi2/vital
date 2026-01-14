//! Common utilities for device-specific implementations

use vitalstream_device_drivers::{DeviceError, Result};

/// Parse ECG lead data from raw bytes
pub fn parse_ecg_leads(data: &[u8], num_leads: usize) -> Result<Vec<Vec<f32>>> {
    if data.len() < num_leads * 2 {
        return Err(DeviceError::InvalidDataFormat(
            "Insufficient data for ECG leads".to_string()
        ));
    }
    
    let mut leads = vec![Vec::new(); num_leads];
    
    for chunk in data.chunks_exact(num_leads * 2) {
        for (i, lead_data) in chunk.chunks_exact(2).enumerate() {
            let value = i16::from_le_bytes([lead_data[0], lead_data[1]]) as f32;
            // Convert to mV (typical ECG range: -5mV to +5mV)
            let mv = value / 1000.0;
            leads[i].push(mv);
        }
    }
    
    Ok(leads)
}

/// Calculate checksum for data validation
pub fn calculate_checksum(data: &[u8]) -> u16 {
    data.iter().fold(0u16, |acc, &byte| acc.wrapping_add(byte as u16))
}

/// Verify checksum
pub fn verify_checksum(data: &[u8], expected: u16) -> Result<()> {
    let calculated = calculate_checksum(data);
    if calculated == expected {
        Ok(())
    } else {
        Err(DeviceError::ChecksumMismatch)
    }
}

/// Parse vital signs from standard format
pub fn parse_vital_signs(data: &[u8]) -> Result<(f32, f32, f32)> {
    if data.len() < 12 {
        return Err(DeviceError::InvalidDataFormat(
            "Insufficient data for vital signs".to_string()
        ));
    }
    
    let hr = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let spo2 = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let rr = f32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    
    Ok((hr, spo2, rr))
}
