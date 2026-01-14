//! GE Healthcare Device Driver
//! 
//! Support for GE CARESCAPE patient monitors

use async_trait::async_trait;
use vitalstream_device_drivers::{
    MedicalDevice, ECGDevice, MultiParameterMonitor,
    DeviceError, DeviceInfo, VitalSign, Result,
    types::{DeviceType, ConnectionType, VitalSignType},
};
use usb_driver::USBDriver;
use chrono::Utc;

pub struct GEMonitor {
    usb: USBDriver,
    device_info: DeviceInfo,
    acquiring: bool,
}

impl GEMonitor {
    pub fn new() -> Result<Self> {
        // GE CARESCAPE typical USB VID/PID
        let usb = USBDriver::new(0x1234, 0x5678)?;
        
        let device_info = DeviceInfo {
            device_id: "GE-CARESCAPE".to_string(),
            manufacturer: "GE Healthcare".to_string(),
            model: "CARESCAPE Monitor".to_string(),
            serial_number: "Unknown".to_string(),
            firmware_version: "Unknown".to_string(),
            hardware_version: "Unknown".to_string(),
            device_type: DeviceType::MultiParameter,
            connection_type: ConnectionType::USB,
            certifications: vec!["FDA 510(k)".to_string(), "CE Mark".to_string()],
        };
        
        Ok(Self {
            usb,
            device_info,
            acquiring: false,
        })
    }
    
    fn parse_ge_data(&self, data: &[u8]) -> Result<Vec<VitalSign>> {
        if data.len() < 20 {
            return Err(DeviceError::InvalidDataFormat(
                "Insufficient GE data".to_string()
            ));
        }
        
        let mut vitals = Vec::new();
        
        // GE uses little-endian format
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
        
        let systolic = f32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        vitals.push(VitalSign {
            sign_type: VitalSignType::SystolicBP,
            value: systolic,
            unit: "mmHg".to_string(),
            timestamp: Utc::now(),
            quality: 0.95,
            patient_id: None,
            metadata: None,
        });
        
        let diastolic = f32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        vitals.push(VitalSign {
            sign_type: VitalSignType::DiastolicBP,
            value: diastolic,
            unit: "mmHg".to_string(),
            timestamp: Utc::now(),
            quality: 0.95,
            patient_id: None,
            metadata: None,
        });
        
        Ok(vitals)
    }
}

#[async_trait]
impl MedicalDevice for GEMonitor {
    async fn get_info(&self) -> Result<DeviceInfo> {
        Ok(self.device_info.clone())
    }
    
    async fn connect(&mut self) -> Result<()> {
        self.usb.connect("GE").await?;
        tracing::info!("GE monitor connected");
        Ok(())
    }
    
    async fn disconnect(&mut self) -> Result<()> {
        self.usb.disconnect().await?;
        tracing::info!("GE monitor disconnected");
        Ok(())
    }
    
    fn is_connected(&self) -> bool {
        self.usb.is_connected()
    }
    
    async fn start_acquisition(&mut self) -> Result<()> {
        let start_cmd = [0x01, 0x53, 0x54, 0x41, 0x52, 0x54]; // "START"
        self.usb.send(&start_cmd).await?;
        self.acquiring = true;
        tracing::info!("Started data acquisition");
        Ok(())
    }
    
    async fn stop_acquisition(&mut self) -> Result<()> {
        let stop_cmd = [0x01, 0x53, 0x54, 0x4F, 0x50]; // "STOP"
        self.usb.send(&stop_cmd).await?;
        self.acquiring = false;
        tracing::info!("Stopped data acquisition");
        Ok(())
    }
    
    async fn read_data(&mut self) -> Result<Vec<VitalSign>> {
        let mut buffer = vec![0u8; 1024];
        let bytes_read = self.usb.receive(&mut buffer).await?;
        
        if bytes_read == 0 {
            return Ok(vec![]);
        }
        
        self.parse_ge_data(&buffer[..bytes_read])
    }
    
    async fn self_test(&mut self) -> Result<bool> {
        let test_cmd = [0x01, 0x54, 0x45, 0x53, 0x54]; // "TEST"
        self.usb.send(&test_cmd).await?;
        
        let mut buffer = vec![0u8; 64];
        let bytes_read = self.usb.receive(&mut buffer).await?;
        
        Ok(bytes_read > 0 && buffer[0] == 0x00) // 0x00 = success
    }
    
    async fn calibrate(&mut self) -> Result<()> {
        let cal_cmd = [0x01, 0x43, 0x41, 0x4C]; // "CAL"
        self.usb.send(&cal_cmd).await?;
        tracing::info!("Calibration initiated");
        Ok(())
    }
}

#[async_trait]
impl ECGDevice for GEMonitor {
    fn get_lead_count(&self) -> u8 {
        12
    }
    
    fn get_sampling_rate(&self) -> u32 {
        500
    }
    
    async fn read_ecg_data(&mut self) -> Result<Vec<Vec<f32>>> {
        let ecg_cmd = [0x01, 0x45, 0x43, 0x47]; // "ECG"
        self.usb.send(&ecg_cmd).await?;
        
        let mut buffer = vec![0u8; 8192];
        let bytes_read = self.usb.receive(&mut buffer).await?;
        
        crate::common::parse_ecg_leads(&buffer[..bytes_read], 12)
    }
    
    async fn set_lead_config(&mut self, leads: u8) -> Result<()> {
        let config_cmd = [0x01, 0x4C, 0x45, 0x41, 0x44, leads];
        self.usb.send(&config_cmd).await?;
        tracing::info!("Set lead configuration to {} leads", leads);
        Ok(())
    }
}

#[async_trait]
impl MultiParameterMonitor for GEMonitor {
    fn get_supported_parameters(&self) -> Vec<String> {
        vec![
            "ECG".to_string(),
            "HR".to_string(),
            "SpO2".to_string(),
            "NIBP".to_string(),
            "RESP".to_string(),
            "TEMP".to_string(),
            "CO2".to_string(),
        ]
    }
    
    async fn set_parameter_enabled(&mut self, parameter: &str, enabled: bool) -> Result<()> {
        let mut cmd = vec![0x01, 0x50, 0x41, 0x52]; // "PAR"
        cmd.extend_from_slice(parameter.as_bytes());
        cmd.push(if enabled { 0x01 } else { 0x00 });
        
        self.usb.send(&cmd).await?;
        tracing::info!("Parameter {} {}", parameter, if enabled { "enabled" } else { "disabled" });
        Ok(())
    }
    
    async fn get_all_vitals(&mut self) -> Result<Vec<VitalSign>> {
        self.read_data().await
    }
}
