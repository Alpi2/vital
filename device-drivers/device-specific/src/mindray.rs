//! Mindray Device Driver
//! 
//! Support for Mindray patient monitors

use async_trait::async_trait;
use vitalstream_device_drivers::{
    MedicalDevice, ECGDevice, MultiParameterMonitor,
    DeviceError, DeviceInfo, VitalSign, Result,
    types::{DeviceType, ConnectionType, VitalSignType},
};
use wifi_driver::WiFiDriver;
use chrono::Utc;

pub struct MindrayMonitor {
    wifi: WiFiDriver,
    device_info: DeviceInfo,
    acquiring: bool,
}

impl MindrayMonitor {
    pub fn new(ip_address: String) -> Result<Self> {
        let wifi = WiFiDriver::new(ip_address)?;
        
        let device_info = DeviceInfo {
            device_id: "MINDRAY-MONITOR".to_string(),
            manufacturer: "Mindray".to_string(),
            model: "BeneView Series".to_string(),
            serial_number: "Unknown".to_string(),
            firmware_version: "Unknown".to_string(),
            hardware_version: "Unknown".to_string(),
            device_type: DeviceType::MultiParameter,
            connection_type: ConnectionType::WiFi,
            certifications: vec!["FDA 510(k)".to_string(), "CE Mark".to_string()],
        };
        
        Ok(Self {
            wifi,
            device_info,
            acquiring: false,
        })
    }
    
    fn parse_mindray_data(&self, data: &[u8]) -> Result<Vec<VitalSign>> {
        if data.len() < 24 {
            return Err(DeviceError::InvalidDataFormat(
                "Insufficient Mindray data".to_string()
            ));
        }
        
        let mut vitals = Vec::new();
        
        // Mindray JSON-like format (simplified)
        let hr = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        vitals.push(VitalSign {
            sign_type: VitalSignType::HeartRate,
            value: hr,
            unit: "bpm".to_string(),
            timestamp: Utc::now(),
            quality: 0.95,
            patient_id: None,
            metadata: None,
        });
        
        let spo2 = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        vitals.push(VitalSign {
            sign_type: VitalSignType::SpO2,
            value: spo2,
            unit: "%".to_string(),
            timestamp: Utc::now(),
            quality: 0.95,
            patient_id: None,
            metadata: None,
        });
        
        let rr = f32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        vitals.push(VitalSign {
            sign_type: VitalSignType::RespirationRate,
            value: rr,
            unit: "br/min".to_string(),
            timestamp: Utc::now(),
            quality: 0.95,
            patient_id: None,
            metadata: None,
        });
        
        let temp = f32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        vitals.push(VitalSign {
            sign_type: VitalSignType::Temperature,
            value: temp,
            unit: "°C".to_string(),
            timestamp: Utc::now(),
            quality: 0.95,
            patient_id: None,
            metadata: None,
        });
        
        Ok(vitals)
    }
}

#[async_trait]
impl MedicalDevice for MindrayMonitor {
    async fn get_info(&self) -> Result<DeviceInfo> {
        Ok(self.device_info.clone())
    }
    
    async fn connect(&mut self) -> Result<()> {
        self.wifi.connect("MINDRAY").await?;
        tracing::info!("Mindray monitor connected");
        Ok(())
    }
    
    async fn disconnect(&mut self) -> Result<()> {
        self.wifi.disconnect().await?;
        tracing::info!("Mindray monitor disconnected");
        Ok(())
    }
    
    fn is_connected(&self) -> bool {
        self.wifi.is_connected()
    }
    
    async fn start_acquisition(&mut self) -> Result<()> {
        let start_cmd = b"{\"cmd\":\"start\"}";
        self.wifi.send(start_cmd).await?;
        self.acquiring = true;
        tracing::info!("Started data acquisition");
        Ok(())
    }
    
    async fn stop_acquisition(&mut self) -> Result<()> {
        let stop_cmd = b"{\"cmd\":\"stop\"}";
        self.wifi.send(stop_cmd).await?;
        self.acquiring = false;
        tracing::info!("Stopped data acquisition");
        Ok(())
    }
    
    async fn read_data(&mut self) -> Result<Vec<VitalSign>> {
        let mut buffer = vec![0u8; 2048];
        let bytes_read = self.wifi.receive(&mut buffer).await?;
        
        if bytes_read == 0 {
            return Ok(vec![]);
        }
        
        self.parse_mindray_data(&buffer[..bytes_read])
    }
    
    async fn self_test(&mut self) -> Result<bool> {
        let test_cmd = b"{\"cmd\":\"test\"}";
        self.wifi.send(test_cmd).await?;
        
        let mut buffer = vec![0u8; 128];
        let bytes_read = self.wifi.receive(&mut buffer).await?;
        
        Ok(bytes_read > 0)
    }
    
    async fn calibrate(&mut self) -> Result<()> {
        let cal_cmd = b"{\"cmd\":\"calibrate\"}";
        self.wifi.send(cal_cmd).await?;
        tracing::info!("Calibration initiated");
        Ok(())
    }
}

#[async_trait]
impl ECGDevice for MindrayMonitor {
    fn get_lead_count(&self) -> u8 {
        12
    }
    
    fn get_sampling_rate(&self) -> u32 {
        500
    }
    
    async fn read_ecg_data(&mut self) -> Result<Vec<Vec<f32>>> {
        let ecg_cmd = b"{\"cmd\":\"ecg\"}";
        self.wifi.send(ecg_cmd).await?;
        
        let mut buffer = vec![0u8; 16384];
        let bytes_read = self.wifi.receive(&mut buffer).await?;
        
        crate::common::parse_ecg_leads(&buffer[..bytes_read], 12)
    }
    
    async fn set_lead_config(&mut self, leads: u8) -> Result<()> {
        let config_cmd = format!("{{\"cmd\":\"set_leads\",\"leads\":{}}}", leads);
        self.wifi.send(config_cmd.as_bytes()).await?;
        tracing::info!("Set lead configuration to {} leads", leads);
        Ok(())
    }
}

#[async_trait]
impl MultiParameterMonitor for MindrayMonitor {
    fn get_supported_parameters(&self) -> Vec<String> {
        vec![
            "ECG".to_string(),
            "HR".to_string(),
            "SpO2".to_string(),
            "NIBP".to_string(),
            "RESP".to_string(),
            "TEMP".to_string(),
            "IBP".to_string(),
            "CO2".to_string(),
        ]
    }
    
    async fn set_parameter_enabled(&mut self, parameter: &str, enabled: bool) -> Result<()> {
        let cmd = format!(
            "{{\"cmd\":\"set_param\",\"param\":\"{}\",\"enabled\":{}}}",
            parameter,
            enabled
        );
        self.wifi.send(cmd.as_bytes()).await?;
        tracing::info!("Parameter {} {}", parameter, if enabled { "enabled" } else { "disabled" });
        Ok(())
    }
    
    async fn get_all_vitals(&mut self) -> Result<Vec<VitalSign>> {
        self.read_data().await
    }
}
