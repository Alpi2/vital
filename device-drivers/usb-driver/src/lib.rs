//! USB Device Driver
//! 
//! FDA/IEC 62304 compliant USB communication for medical devices

use async_trait::async_trait;
use rusb::{Context, Device, DeviceHandle, UsbContext};
use std::time::Duration;

use vitalstream_device_drivers::{
    DeviceConnection, DeviceError, DeviceInfo, Result,
    types::{ConnectionType, DeviceType},
};

pub struct USBDriver {
    context: Context,
    device_handle: Option<DeviceHandle<Context>>,
    connected: bool,
    vendor_id: u16,
    product_id: u16,
}

impl USBDriver {
    pub fn new(vendor_id: u16, product_id: u16) -> Result<Self> {
        let context = Context::new()
            .map_err(|e| DeviceError::HardwareError(e.to_string()))?;
        
        Ok(Self {
            context,
            device_handle: None,
            connected: false,
            vendor_id,
            product_id,
        })
    }
    
    fn find_device(&self) -> Result<Device<Context>> {
        let devices = self.context.devices()
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        
        for device in devices.iter() {
            let device_desc = device.device_descriptor()
                .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
            
            if device_desc.vendor_id() == self.vendor_id 
                && device_desc.product_id() == self.product_id {
                return Ok(device);
            }
        }
        
        Err(DeviceError::DeviceNotFound(format!(
            "USB device {:04x}:{:04x} not found",
            self.vendor_id, self.product_id
        )))
    }
    
    fn get_device_info(&self, device: &Device<Context>) -> Result<DeviceInfo> {
        let device_desc = device.device_descriptor()
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        
        let handle = device.open()
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        
        let timeout = Duration::from_secs(1);
        let languages = handle.read_languages(timeout)
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        
        let language = languages.get(0)
            .ok_or_else(|| DeviceError::ProtocolError("No language found".to_string()))?;
        
        let manufacturer = handle
            .read_manufacturer_string(*language, &device_desc, timeout)
            .unwrap_or_else(|_| "Unknown".to_string());
        
        let product = handle
            .read_product_string(*language, &device_desc, timeout)
            .unwrap_or_else(|_| "Unknown".to_string());
        
        let serial = handle
            .read_serial_number_string(*language, &device_desc, timeout)
            .unwrap_or_else(|_| "Unknown".to_string());
        
        Ok(DeviceInfo {
            device_id: format!("USB:{:04x}:{:04x}", self.vendor_id, self.product_id),
            manufacturer,
            model: product,
            serial_number: serial,
            firmware_version: "Unknown".to_string(),
            hardware_version: format!("{}.{}", device_desc.device_version().major(), device_desc.device_version().minor()),
            device_type: DeviceType::Unknown,
            connection_type: ConnectionType::USB,
            certifications: vec![],
        })
    }
}

#[async_trait]
impl DeviceConnection for USBDriver {
    async fn discover(&self) -> Result<Vec<DeviceInfo>> {
        tracing::info!("Discovering USB devices...");
        
        let devices = self.context.devices()
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        
        let mut device_infos = Vec::new();
        
        for device in devices.iter() {
            if let Ok(info) = self.get_device_info(&device) {
                device_infos.push(info);
            }
        }
        
        Ok(device_infos)
    }
    
    async fn connect(&mut self, _device_id: &str) -> Result<()> {
        tracing::info!("Connecting to USB device {:04x}:{:04x}", self.vendor_id, self.product_id);
        
        let device = self.find_device()?;
        
        let mut handle = device.open()
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        
        // Claim interface 0
        handle.claim_interface(0)
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        
        self.device_handle = Some(handle);
        self.connected = true;
        
        tracing::info!("Successfully connected to USB device");
        Ok(())
    }
    
    async fn disconnect(&mut self) -> Result<()> {
        if let Some(handle) = &mut self.device_handle {
            handle.release_interface(0)
                .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        }
        
        self.device_handle = None;
        self.connected = false;
        
        tracing::info!("Disconnected from USB device");
        Ok(())
    }
    
    async fn send(&mut self, data: &[u8]) -> Result<usize> {
        let handle = self.device_handle.as_ref()
            .ok_or_else(|| DeviceError::DeviceNotFound("Not connected".to_string()))?;
        
        let timeout = Duration::from_secs(1);
        let endpoint = 0x01; // Bulk OUT endpoint
        
        let written = handle.write_bulk(endpoint, data, timeout)
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        
        Ok(written)
    }
    
    async fn receive(&mut self, buffer: &mut [u8]) -> Result<usize> {
        let handle = self.device_handle.as_ref()
            .ok_or_else(|| DeviceError::DeviceNotFound("Not connected".to_string()))?;
        
        let timeout = Duration::from_secs(1);
        let endpoint = 0x81; // Bulk IN endpoint
        
        let read = handle.read_bulk(endpoint, buffer, timeout)
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
    fn test_usb_driver_creation() {
        let result = USBDriver::new(0x1234, 0x5678);
        assert!(result.is_ok());
    }
}
