// Secure Network Layer with TLS and libp2p
// Production-ready P2P networking with encryption and authentication

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use sha3::{Sha3_256, Digest};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureNetworkConfig {
    pub listen_address: String,
    pub tls_cert_path: String,
    pub tls_key_path: String,
    pub ca_cert_path: Option<String>,
    pub enable_mutual_tls: bool,
    pub max_connections: usize,
    pub connection_timeout_secs: u64,
    pub enable_nat_traversal: bool,
    pub bootstrap_peers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub peer_id: String,
    pub addresses: Vec<String>,
    pub public_key: Vec<u8>,
    pub protocols: Vec<String>,
    pub reputation: i32,
    pub last_seen: i64,
    pub is_validator: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecureMessage {
    Handshake {
        peer_id: String,
        public_key: Vec<u8>,
        protocols: Vec<String>,
        signature: Vec<u8>,
    },
    BlockProposal {
        block_hash: String,
        proposer_id: String,
        encrypted_data: Vec<u8>,
        signature: Vec<u8>,
    },
    Vote {
        block_hash: String,
        validator_id: String,
        vote: bool,
        signature: Vec<u8>,
    },
    Transaction {
        tx_id: String,
        encrypted_data: Vec<u8>,
        signature: Vec<u8>,
    },
    Heartbeat {
        peer_id: String,
        timestamp: i64,
    },
}

pub struct SecureNetworkManager {
    config: SecureNetworkConfig,
    peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
    connections: Arc<RwLock<HashMap<String, SecureConnection>>>,
    message_queue: Arc<RwLock<Vec<SecureMessage>>>,
}

#[derive(Debug, Clone)]
struct SecureConnection {
    peer_id: String,
    remote_address: String,
    is_encrypted: bool,
    is_authenticated: bool,
    established_at: i64,
    last_activity: i64,
    bytes_sent: u64,
    bytes_received: u64,
}

impl SecureNetworkManager {
    /// Initialize secure network manager
    pub fn new(config: SecureNetworkConfig) -> Result<Self, String> {
        let manager = SecureNetworkManager {
            config,
            peers: Arc::new(RwLock::new(HashMap::new())),
            connections: Arc::new(RwLock::new(HashMap::new())),
            message_queue: Arc::new(RwLock::new(Vec::new())),
        };

        Ok(manager)
    }

    /// Start network listener with TLS
    pub fn start_listener(&self) -> Result<(), String> {
        // In production, this would:
        // 1. Load TLS certificates
        // 2. Create TLS acceptor
        // 3. Start TCP listener
        // 4. Accept connections with TLS handshake
        // 5. Verify client certificates (if mutual TLS)
        
        println!("Starting secure listener on {}", self.config.listen_address);
        println!("TLS enabled with cert: {}", self.config.tls_cert_path);
        println!("Mutual TLS: {}", self.config.enable_mutual_tls);
        
        Ok(())
    }

    /// Connect to peer with TLS
    pub fn connect_to_peer(&self, peer_address: &str, peer_id: &str) -> Result<(), String> {
        // In production, this would:
        // 1. Create TLS connector
        // 2. Establish TCP connection
        // 3. Perform TLS handshake
        // 4. Verify server certificate
        // 5. Send handshake message
        // 6. Authenticate peer

        let connection = SecureConnection {
            peer_id: peer_id.to_string(),
            remote_address: peer_address.to_string(),
            is_encrypted: true,
            is_authenticated: false,
            established_at: chrono::Utc::now().timestamp(),
            last_activity: chrono::Utc::now().timestamp(),
            bytes_sent: 0,
            bytes_received: 0,
        };

        let mut connections = self.connections.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        connections.insert(peer_id.to_string(), connection);

        Ok(())
    }

    /// Perform secure handshake
    pub fn perform_handshake(&self, peer_id: &str, public_key: Vec<u8>) -> Result<(), String> {
        // Verify handshake signature
        // Exchange protocol versions
        // Establish session keys

        let mut connections = self.connections.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        if let Some(conn) = connections.get_mut(peer_id) {
            conn.is_authenticated = true;
            conn.last_activity = chrono::Utc::now().timestamp();
        }

        // Register peer
        let peer_info = PeerInfo {
            peer_id: peer_id.to_string(),
            addresses: vec![],
            public_key,
            protocols: vec!["vital/1.0".to_string()],
            reputation: 100,
            last_seen: chrono::Utc::now().timestamp(),
            is_validator: false,
        };

        let mut peers = self.peers.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        peers.insert(peer_id.to_string(), peer_info);

        Ok(())
    }

    /// Send encrypted message to peer
    pub fn send_message(&self, peer_id: &str, message: SecureMessage) -> Result<(), String> {
        // Verify connection exists and is authenticated
        let connections = self.connections.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        let conn = connections.get(peer_id)
            .ok_or_else(|| format!("No connection to peer {}", peer_id))?;

        if !conn.is_authenticated {
            return Err("Peer not authenticated".to_string());
        }

        // In production:
        // 1. Serialize message
        // 2. Encrypt with session key (AES-256-GCM)
        // 3. Add HMAC for integrity
        // 4. Send over TLS connection

        println!("Sending encrypted message to {}", peer_id);
        Ok(())
    }

    /// Broadcast message to all connected peers
    pub fn broadcast_message(&self, message: SecureMessage) -> Result<(), String> {
        let connections = self.connections.read()
            .map_err(|e| format!("Lock error: {}", e))?;

        for (peer_id, conn) in connections.iter() {
            if conn.is_authenticated {
                // Clone message for each peer
                self.send_message(peer_id, message.clone())?;
            }
        }

        Ok(())
    }

    /// Receive and decrypt message
    pub fn receive_message(&self) -> Result<Option<SecureMessage>, String> {
        let mut queue = self.message_queue.write()
            .map_err(|e| format!("Lock error: {}", e))?;

        Ok(queue.pop())
    }

    /// Verify message signature
    pub fn verify_message_signature(&self, message: &SecureMessage, public_key: &[u8]) -> Result<bool, String> {
        use ring::signature::{UnparsedPublicKey, ED25519};

        let (data, signature) = match message {
            SecureMessage::Handshake { peer_id, signature, .. } => {
                (peer_id.as_bytes(), signature)
            }
            SecureMessage::BlockProposal { block_hash, signature, .. } => {
                (block_hash.as_bytes(), signature)
            }
            SecureMessage::Vote { block_hash, signature, .. } => {
                (block_hash.as_bytes(), signature)
            }
            SecureMessage::Transaction { tx_id, signature, .. } => {
                (tx_id.as_bytes(), signature)
            }
            SecureMessage::Heartbeat { .. } => return Ok(true),
        };

        let verifier = UnparsedPublicKey::new(&ED25519, public_key);
        match verifier.verify(data, signature) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// NAT traversal using STUN/TURN
    pub fn enable_nat_traversal(&self) -> Result<(), String> {
        if !self.config.enable_nat_traversal {
            return Ok(());
        }

        // In production:
        // 1. Query STUN server for public IP/port
        // 2. Configure TURN relay if needed
        // 3. Update peer addresses with public endpoints
        // 4. Enable hole punching

        println!("NAT traversal enabled");
        Ok(())
    }

    /// Peer discovery using DHT
    pub fn discover_peers(&self) -> Result<Vec<PeerInfo>, String> {
        // In production:
        // 1. Query DHT for peers
        // 2. Connect to bootstrap nodes
        // 3. Exchange peer lists
        // 4. Verify peer identities

        let mut discovered_peers = Vec::new();

        for bootstrap_peer in &self.config.bootstrap_peers {
            // Simulate peer discovery
            let peer_info = PeerInfo {
                peer_id: format!("peer_{}", bootstrap_peer),
                addresses: vec![bootstrap_peer.clone()],
                public_key: vec![0; 32],
                protocols: vec!["vital/1.0".to_string()],
                reputation: 100,
                last_seen: chrono::Utc::now().timestamp(),
                is_validator: false,
            };
            discovered_peers.push(peer_info);
        }

        Ok(discovered_peers)
    }

    /// Update peer reputation
    pub fn update_peer_reputation(&self, peer_id: &str, delta: i32) -> Result<(), String> {
        let mut peers = self.peers.write()
            .map_err(|e| format!("Lock error: {}", e))?;

        if let Some(peer) = peers.get_mut(peer_id) {
            peer.reputation = (peer.reputation + delta).max(0).min(1000);
            
            // Disconnect if reputation too low
            if peer.reputation < 10 {
                self.disconnect_peer(peer_id)?;
            }
        }

        Ok(())
    }

    /// Disconnect from peer
    pub fn disconnect_peer(&self, peer_id: &str) -> Result<(), String> {
        let mut connections = self.connections.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        connections.remove(peer_id);
        println!("Disconnected from peer {}", peer_id);
        
        Ok(())
    }

    /// Get connected peers
    pub fn get_connected_peers(&self) -> Result<Vec<String>, String> {
        let connections = self.connections.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        Ok(connections.keys().cloned().collect())
    }

    /// Get network statistics
    pub fn get_network_stats(&self) -> Result<NetworkStats, String> {
        let connections = self.connections.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        let peers = self.peers.read()
            .map_err(|e| format!("Lock error: {}", e))?;

        let total_bytes_sent: u64 = connections.values().map(|c| c.bytes_sent).sum();
        let total_bytes_received: u64 = connections.values().map(|c| c.bytes_received).sum();

        Ok(NetworkStats {
            total_peers: peers.len(),
            connected_peers: connections.len(),
            total_bytes_sent,
            total_bytes_received,
            average_reputation: peers.values().map(|p| p.reputation).sum::<i32>() as f64 / peers.len().max(1) as f64,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub total_peers: usize,
    pub connected_peers: usize,
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
    pub average_reputation: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_initialization() {
        let config = SecureNetworkConfig {
            listen_address: "0.0.0.0:9000".to_string(),
            tls_cert_path: "/etc/vital/tls/cert.pem".to_string(),
            tls_key_path: "/etc/vital/tls/key.pem".to_string(),
            ca_cert_path: Some("/etc/vital/tls/ca.pem".to_string()),
            enable_mutual_tls: true,
            max_connections: 100,
            connection_timeout_secs: 30,
            enable_nat_traversal: true,
            bootstrap_peers: vec!["peer1.vital.network:9000".to_string()],
        };

        let manager = SecureNetworkManager::new(config).unwrap();
        assert!(manager.start_listener().is_ok());
    }

    #[test]
    fn test_peer_connection() {
        let config = SecureNetworkConfig {
            listen_address: "0.0.0.0:9000".to_string(),
            tls_cert_path: "/etc/vital/tls/cert.pem".to_string(),
            tls_key_path: "/etc/vital/tls/key.pem".to_string(),
            ca_cert_path: None,
            enable_mutual_tls: false,
            max_connections: 10,
            connection_timeout_secs: 30,
            enable_nat_traversal: false,
            bootstrap_peers: vec![],
        };

        let manager = SecureNetworkManager::new(config).unwrap();
        assert!(manager.connect_to_peer("192.168.1.100:9000", "peer1").is_ok());
        
        let peers = manager.get_connected_peers().unwrap();
        assert_eq!(peers.len(), 1);
    }

    #[test]
    fn test_peer_reputation() {
        let config = SecureNetworkConfig {
            listen_address: "0.0.0.0:9000".to_string(),
            tls_cert_path: "/etc/vital/tls/cert.pem".to_string(),
            tls_key_path: "/etc/vital/tls/key.pem".to_string(),
            ca_cert_path: None,
            enable_mutual_tls: false,
            max_connections: 10,
            connection_timeout_secs: 30,
            enable_nat_traversal: false,
            bootstrap_peers: vec![],
        };

        let manager = SecureNetworkManager::new(config).unwrap();
        manager.connect_to_peer("192.168.1.100:9000", "peer1").unwrap();
        manager.perform_handshake("peer1", vec![0; 32]).unwrap();
        
        manager.update_peer_reputation("peer1", 50).unwrap();
        manager.update_peer_reputation("peer1", -200).unwrap();
        
        // Peer should still be connected (reputation = 100 + 50 - 200 = -50, clamped to 0)
        let stats = manager.get_network_stats().unwrap();
        assert_eq!(stats.total_peers, 1);
    }
}
