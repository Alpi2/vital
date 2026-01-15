//! Network Latency Optimization
//! Reduces network latency for real-time vital signs streaming

use tokio::net::TcpStream;
use tokio_tungstenite::{WebSocketStream, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::Write;

#[derive(Debug, Serialize, Deserialize)]
pub struct VitalSignsPacket {
    pub patient_id: i32,
    pub timestamp: i64,
    pub heart_rate: i32,
    pub spo2: i32,
    pub respiratory_rate: i32,
}

/// WebSocket optimizer with compression and batching
pub struct WebSocketOptimizer {
    compression_enabled: bool,
    batch_size: usize,
    batch_timeout_ms: u64,
}

impl WebSocketOptimizer {
    pub fn new() -> Self {
        Self {
            compression_enabled: true,
            batch_size: 10,
            batch_timeout_ms: 50,
        }
    }

    /// Send data with compression
    pub async fn send_compressed(
        &self,
        ws: &mut WebSocketStream<TcpStream>,
        data: &VitalSignsPacket,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(data)?;
        
        if self.compression_enabled && json.len() > 1024 {
            // Compress large payloads
            let compressed = self.compress_data(json.as_bytes())?;
            ws.send(Message::Binary(compressed)).await?;
        } else {
            // Send small payloads uncompressed
            ws.send(Message::Text(json)).await?;
        }
        
        Ok(())
    }

    /// Batch multiple packets for efficient transmission
    pub async fn send_batch(
        &self,
        ws: &mut WebSocketStream<TcpStream>,
        packets: Vec<VitalSignsPacket>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(&packets)?;
        let compressed = self.compress_data(json.as_bytes())?;
        ws.send(Message::Binary(compressed)).await?;
        Ok(())
    }

    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(data)?;
        encoder.finish()
    }
}

/// HTTP/2 connection pooling for REST APIs
pub struct Http2ConnectionPool {
    client: reqwest::Client,
}

impl Http2ConnectionPool {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .http2_prior_knowledge()
            .pool_max_idle_per_host(10)
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            .tcp_keepalive(std::time::Duration::from_secs(60))
            .tcp_nodelay(true)
            .build()
            .unwrap();
        
        Self { client }
    }

    pub async fn post_vitals(
        &self,
        url: &str,
        data: &VitalSignsPacket,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let start = Instant::now();
        
        self.client
            .post(url)
            .json(data)
            .send()
            .await?;
        
        let latency = start.elapsed().as_millis();
        println!("HTTP/2 POST latency: {}ms", latency);
        
        Ok(())
    }
}

/// UDP-based low-latency streaming (for non-critical data)
pub struct UdpStreamer {
    socket: tokio::net::UdpSocket,
}

impl UdpStreamer {
    pub async fn new(bind_addr: &str) -> Result<Self, std::io::Error> {
        let socket = tokio::net::UdpSocket::bind(bind_addr).await?;
        Ok(Self { socket })
    }

    pub async fn send_vitals(
        &self,
        target: &str,
        data: &VitalSignsPacket,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(data)?;
        self.socket.send_to(json.as_bytes(), target).await?;
        Ok(())
    }
}

/// Network performance monitor
pub struct NetworkMonitor {
    latencies: Vec<u64>,
}

impl NetworkMonitor {
    pub fn new() -> Self {
        Self {
            latencies: Vec::with_capacity(1000),
        }
    }

    pub fn record_latency(&mut self, latency_ms: u64) {
        if self.latencies.len() >= 1000 {
            self.latencies.remove(0);
        }
        self.latencies.push(latency_ms);
    }

    pub fn get_stats(&self) -> NetworkStats {
        if self.latencies.is_empty() {
            return NetworkStats::default();
        }

        let sum: u64 = self.latencies.iter().sum();
        let avg = sum as f64 / self.latencies.len() as f64;
        
        let mut sorted = self.latencies.clone();
        sorted.sort_unstable();
        let p50 = sorted[sorted.len() / 2];
        let p95 = sorted[(sorted.len() * 95) / 100];
        let p99 = sorted[(sorted.len() * 99) / 100];

        NetworkStats {
            avg_latency_ms: avg,
            p50_latency_ms: p50,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
            min_latency_ms: *sorted.first().unwrap(),
            max_latency_ms: *sorted.last().unwrap(),
        }
    }
}

#[derive(Debug, Default, Serialize)]
pub struct NetworkStats {
    pub avg_latency_ms: f64,
    pub p50_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub p99_latency_ms: u64,
    pub min_latency_ms: u64,
    pub max_latency_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_monitor() {
        let mut monitor = NetworkMonitor::new();
        
        for i in 1..=100 {
            monitor.record_latency(i);
        }
        
        let stats = monitor.get_stats();
        assert!(stats.avg_latency_ms > 0.0);
        assert!(stats.p99_latency_ms >= stats.p95_latency_ms);
        assert!(stats.p95_latency_ms >= stats.p50_latency_ms);
    }
}
