//! IEEE 11073 Protocol Implementation
//! 
//! Personal Health Device (PHD) communication standard

use crate::{DeviceError, Result};
use serde::{Deserialize, Serialize};

/// IEEE 11073 Protocol Handler
pub struct IEEE11073Protocol {
    protocol_version: u16,
}

impl Default for IEEE11073Protocol {
    fn default() -> Self {
        Self {
            protocol_version: 0x8000, // IEEE 11073-20601
        }
    }
}

impl IEEE11073Protocol {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Parse IEEE 11073 APDU (Application Protocol Data Unit)
    pub fn parse_apdu(&self, data: &[u8]) -> Result<APDU> {
        if data.len() < 4 {
            return Err(DeviceError::InvalidDataFormat(
                "APDU too short".to_string()
            ));
        }
        
        let choice = u16::from_be_bytes([data[0], data[1]]);
        let length = u16::from_be_bytes([data[2], data[3]]);
        
        Ok(APDU {
            choice,
            length,
            value: data[4..].to_vec(),
        })
    }
    
    /// Create association request
    pub fn create_association_request(&self) -> Vec<u8> {
        // Simplified IEEE 11073 association request
        let mut apdu = vec![
            0xE2, 0x00, // APDU choice: Association Request
            0x00, 0x32, // Length
        ];
        
        // Protocol version
        apdu.extend_from_slice(&self.protocol_version.to_be_bytes());
        
        // Add more fields as needed
        apdu
    }
    
    /// Parse measurement data
    pub fn parse_measurement(&self, data: &[u8]) -> Result<Measurement> {
        // Simplified measurement parsing
        if data.len() < 8 {
            return Err(DeviceError::InvalidDataFormat(
                "Measurement data too short".to_string()
            ));
        }
        
        let obj_handle = u16::from_be_bytes([data[0], data[1]]);
        let attribute_id = u16::from_be_bytes([data[2], data[3]]);
        let value = i32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        
        Ok(Measurement {
            obj_handle,
            attribute_id,
            value: value as f32,
            unit: self.get_unit_from_attribute(attribute_id),
        })
    }
    
    fn get_unit_from_attribute(&self, attribute_id: u16) -> String {
        match attribute_id {
            0x0A46 => "bpm".to_string(),      // Heart rate
            0x0A4C => "%".to_string(),        // SpO2
            0x0A80 => "mmHg".to_string(),     // Blood pressure
            _ => "unknown".to_string(),
        }
    }
}

/// APDU structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APDU {
    pub choice: u16,
    pub length: u16,
    pub value: Vec<u8>,
}

/// Measurement data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    pub obj_handle: u16,
    pub attribute_id: u16,
    pub value: f32,
    pub unit: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_apdu() {
        let protocol = IEEE11073Protocol::new();
        let data = vec![0xE2, 0x00, 0x00, 0x04, 0x01, 0x02, 0x03, 0x04];
        
        let result = protocol.parse_apdu(&data);
        assert!(result.is_ok());
        
        let apdu = result.unwrap();
        assert_eq!(apdu.choice, 0xE200);
        assert_eq!(apdu.length, 4);
    }
}
