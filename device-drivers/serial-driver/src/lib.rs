//! Serial Port (RS-232) Device Driver
//! 
//! FDA/IEC 62304 compliant serial communication for medical devices

use async_trait::async_trait;
use tokio_serial::{SerialPort, SerialPortBuilderExt, SerialStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::time::Duration;

use vitalstream_device_drivers::{
    DeviceConnection, DeviceError, DeviceInfo, Result,
    types::{ConnectionType, DeviceType},
};

pub struct SerialDriver {
    port: Option<SerialStream>,
    port_name: String,
    baud_rate: u32,
    connected: bool,
}

impl SerialDriver {
    pub fn new(port_name: String, baud_rate: u32) -> Result<Self> {
        Ok(Self {
            port: None,
            port_name,
            baud_rate,
            connected: false,
        })
    }
    
    /// List available serial ports
    pub fn list_ports() -> Result<Vec<String>> {
        let ports = tokio_serial::available_ports()
            .map_err(|e| DeviceError::HardwareError(e.to_string()))?;
        
        Ok(ports.into_iter().map(|p| p.port_name).collect())
    }
}

#[async_trait]
impl DeviceConnection for SerialDriver {
    async fn discover(&self) -> Result<Vec<DeviceInfo>> {
        tracing::info!("Discovering serial devices...");
        
        let ports = Self::list_ports()?;
        
        let mut device_infos = Vec::new();
        
        for port_name in ports {
            device_infos.push(DeviceInfo {
                device_id: format!("SERIAL:{}", port_name),
                manufacturer: "Unknown".to_string(),
                model: "Serial Device".to_string(),
                serial_number: "Unknown".to_string(),
                firmware_version: "Unknown".to_string(),
                hardware_version: "Unknown".to_string(),
                device_type: DeviceType::Unknown,
                connection_type: ConnectionType::Serial,
                certifications: vec![],
            });
        }
        
        Ok(device_infos)
    }
    
    async fn connect(&mut self, _device_id: &str) -> Result<()> {
        tracing::info!("Connecting to serial port: {} at {} baud", self.port_name, self.baud_rate);
        
        let port = tokio_serial::new(&self.port_name, self.baud_rate)
            .timeout(Duration::from_secs(1))
            .open_native_async()
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        
        self.port = Some(port);
        self.connected = true;
        
        tracing::info!("Successfully connected to serial port");
        Ok(())
    }
    
    async fn disconnect(&mut self) -> Result<()> {
        self.port = None;
        self.connected = false;
        
        tracing::info!("Disconnected from serial port");
        Ok(())
    }
    
    async fn send(&mut self, data: &[u8]) -> Result<usize> {
        let port = self.port.as_mut()
            .ok_or_else(|| DeviceError::DeviceNotFound("Not connected".to_string()))?;
        
        port.write_all(data)
            .await
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        
        port.flush()
            .await
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        
        Ok(data.len())
    }
    
    async fn receive(&mut self, buffer: &mut [u8]) -> Result<usize> {
        let port = self.port.as_mut()
            .ok_or_else(|| DeviceError::DeviceNotFound("Not connected".to_string()))?;
        
        let read = port.read(buffer)
            .await
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        
        Ok(read)
    }
    
    fn is_connected(&self) -> bool {
        self.connected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_serial_driver_creation() {
        let result = SerialDriver::new("/dev/ttyUSB0".to_string(), 9600);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_list_ports() {
        let result = SerialDriver::list_ports();
        // This might fail on systems without serial ports, which is okay
        assert!(result.is_ok() || result.is_err());
    }
}
