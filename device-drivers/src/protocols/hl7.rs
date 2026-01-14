//! HL7 Protocol Parser
//! 
//! Implements HL7 v2.x message parsing for medical device data
//! Note: For production, integrate with Java HAPI FHIR service

use crate::{DeviceError, Result};
use serde::{Deserialize, Serialize};

/// HL7 Message Parser
pub struct HL7Parser {
    field_separator: char,
    component_separator: char,
    repetition_separator: char,
    escape_character: char,
    subcomponent_separator: char,
}

impl Default for HL7Parser {
    fn default() -> Self {
        Self {
            field_separator: '|',
            component_separator: '^',
            repetition_separator: '~',
            escape_character: '\\',
            subcomponent_separator: '&',
        }
    }
}

impl HL7Parser {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Parse HL7 message
    pub fn parse(&self, message: &str) -> Result<HL7Message> {
        let segments: Vec<&str> = message.split('\r').collect();
        
        if segments.is_empty() {
            return Err(DeviceError::InvalidDataFormat(
                "Empty HL7 message".to_string()
            ));
        }
        
        // Parse MSH (Message Header) segment
        let msh = self.parse_msh(segments[0])?;
        
        Ok(HL7Message {
            message_type: msh.message_type.clone(),
            message_control_id: msh.message_control_id.clone(),
            segments: segments.iter().map(|s| s.to_string()).collect(),
            raw: message.to_string(),
        })
    }
    
    /// Parse MSH segment
    fn parse_msh(&self, segment: &str) -> Result<MSHSegment> {
        let fields: Vec<&str> = segment.split(self.field_separator).collect();
        
        if fields.len() < 12 || fields[0] != "MSH" {
            return Err(DeviceError::InvalidDataFormat(
                "Invalid MSH segment".to_string()
            ));
        }
        
        Ok(MSHSegment {
            message_type: fields[8].to_string(),
            message_control_id: fields[9].to_string(),
            version: fields[11].to_string(),
        })
    }
    
    /// Create ORU (Observation Result) message for vital signs
    pub fn create_oru_message(
        &self,
        patient_id: &str,
        observations: Vec<Observation>,
    ) -> Result<String> {
        let timestamp = chrono::Utc::now().format("%Y%m%d%H%M%S").to_string();
        let control_id = uuid::Uuid::new_v4().to_string();
        
        let mut message = String::new();
        
        // MSH segment
        message.push_str(&format!(
            "MSH|^~\\&|VitalStream|Hospital|HIS|Hospital|{}||ORU^R01|{}|P|2.5\r",
            timestamp, control_id
        ));
        
        // PID segment
        message.push_str(&format!(
            "PID|1||{}||||||||||||||\r",
            patient_id
        ));
        
        // OBR segment
        message.push_str(&format!(
            "OBR|1|||VITALS^Vital Signs|||{}\r",
            timestamp
        ));
        
        // OBX segments for each observation
        for (idx, obs) in observations.iter().enumerate() {
            message.push_str(&format!(
                "OBX|{}|NM|{}^{}||{}|{}||||F\r",
                idx + 1,
                obs.code,
                obs.name,
                obs.value,
                obs.unit
            ));
        }
        
        Ok(message)
    }
}

/// HL7 Message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HL7Message {
    pub message_type: String,
    pub message_control_id: String,
    pub segments: Vec<String>,
    pub raw: String,
}

/// MSH (Message Header) segment
#[derive(Debug, Clone)]
struct MSHSegment {
    message_type: String,
    message_control_id: String,
    version: String,
}

/// Observation data for HL7 messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub code: String,
    pub name: String,
    pub value: String,
    pub unit: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hl7_parser() {
        let parser = HL7Parser::new();
        let message = "MSH|^~\\&|VitalStream|Hospital|HIS|Hospital|20260103120000||ORU^R01|12345|P|2.5\r";
        
        let result = parser.parse(message);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_create_oru() {
        let parser = HL7Parser::new();
        let observations = vec![
            Observation {
                code: "HR".to_string(),
                name: "Heart Rate".to_string(),
                value: "75".to_string(),
                unit: "bpm".to_string(),
            },
        ];
        
        let result = parser.create_oru_message("PAT001", observations);
        assert!(result.is_ok());
    }
}
