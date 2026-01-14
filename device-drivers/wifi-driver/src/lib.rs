//! Wi-Fi Device Driver
//! 
//! FDA/IEC 62304 compliant Wi-Fi communication for medical devices
//! Supports mDNS/Bonjour discovery and WebSocket communication

use async_trait::async_trait;
use mdns_sd::{ServiceDaemon, ServiceEvent};
use std::time::Duration;
use tokio::time;
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::Message};

use vitalstream_device_drivers::{
    DeviceConnection, DeviceError, DeviceInfo, Result,
    types::{ConnectionType, DeviceType},
};

pub struct WiFiDriver {
    device_url: String,
    ws_stream: Option<tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>>,
    connected: bool,
}

impl WiFiDriver {
    pub fn new(device_url: String) -> Result<Self> {
        Ok(Self {
            device_url,
            ws_stream: None,
            connected: false,
        })
    }
    
    /// Discover devices using mDNS/Bonjour
    pub async fn discover_mdns(service_type: &str) -> Result<Vec<DeviceInfo>> {
        tracing::info!("Starting mDNS discovery for: {}", service_type);
        
        let mdns = ServiceDaemon::new()
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        
        let receiver = mdns.browse(service_type)
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        
        let mut devices = Vec::new();
        let timeout = time::sleep(Duration::from_secs(5));
        tokio::pin!(timeout);
        
        loop {
            tokio::select! {
                event = receiver.recv_async() => {
                    match event {
                        Ok(ServiceEvent::ServiceResolved(info)) => {
                            let device_info = DeviceInfo {
                                device_id: format!("WIFI:{}", info.get_fullname()),
                                manufacturer: "Unknown".to_string(),
                                model: info.get_type().to_string(),
                                serial_number: "Unknown".to_string(),
                                firmware_version: "Unknown".to_string(),
                                hardware_version: "Unknown".to_string(),
                                device_type: DeviceType::Unknown,
                                connection_type: ConnectionType::WiFi,
                                certifications: vec![],
                            };
                            devices.push(device_info);
                        }
                        _ => {}
                    }
                }
                _ = &mut timeout => {
                    break;
                }
            }
        }
        
        tracing::info!("Found {} devices via mDNS", devices.len());
        Ok(devices)
    }
}

#[async_trait]
impl DeviceConnection for WiFiDriver {
    async fn discover(&self) -> Result<Vec<DeviceInfo>> {
        // Discover medical devices on network
        Self::discover_mdns("_medical-device._tcp.local.").await
    }
    
    async fn connect(&mut self, device_id: &str) -> Result<()> {
        tracing::info!("Connecting to Wi-Fi device: {}", device_id);
        
        // Connect via WebSocket
        let url = format!("ws://{}/ws", self.device_url);
        
        let (ws_stream, _) = connect_async(&url)
            .await
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        
        self.ws_stream = Some(ws_stream);
        self.connected = true;
        
        tracing::info!("Successfully connected to Wi-Fi device");
        Ok(())
    }
    
    async fn disconnect(&mut self) -> Result<()> {
        if let Some(mut ws) = self.ws_stream.take() {
            ws.close(None)
                .await
                .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        }
        
        self.connected = false;
        tracing::info!("Disconnected from Wi-Fi device");
        Ok(())
    }
    
    async fn send(&mut self, data: &[u8]) -> Result<usize> {
        let ws = self.ws_stream.as_mut()
            .ok_or_else(|| DeviceError::DeviceNotFound("Not connected".to_string()))?;
        
        ws.send(Message::Binary(data.to_vec()))
            .await
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;
        
        Ok(data.len())
    }
    
    async fn receive(&mut self, buffer: &mut [u8]) -> Result<usize> {
        let ws = self.ws_stream.as_mut()
            .ok_or_else(|| DeviceError::DeviceNotFound("Not connected".to_string()))?;
        
        match ws.next().await {
            Some(Ok(Message::Binary(data))) => {
                let len = data.len().min(buffer.len());
                buffer[..len].copy_from_slice(&data[..len]);
                Ok(len)
            }
            Some(Ok(_)) => Ok(0),
            Some(Err(e)) => Err(DeviceError::ConnectionFailed(e.to_string())),
            None => Err(DeviceError::DeviceDisconnected("Connection closed".to_string())),
        }
    }
    
    fn is_connected(&self) -> bool {
        self.connected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_wifi_driver_creation() {
        let result = WiFiDriver::new("192.168.1.100:8080".to_string());
        assert!(result.is_ok());
    }
}
