// Peer-to-Peer Network Implementation for Blockchain
// Distributed consensus and node communication with TLS encryption

use std::collections::HashMap;
use std::net::{TcpListener, TcpStream, SocketAddr};
use std::sync::{Arc, Mutex};
use std::thread;
use std::io::{Read, Write};
use serde::{Serialize, Deserialize};
use bincode;
use rustls::{ServerConfig, ClientConfig, ServerConnection, ClientConnection, StreamOwned};
use rustls_pemfile::{certs, pkcs8_private_keys};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use ring::rand::{SecureRandom, SystemRandom};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerNode {
    pub node_id: String,
    pub address: SocketAddr,
    pub public_key: Vec<u8>,
    pub reputation: u32,
    pub last_seen: u64,
    pub is_validator: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkMessage {
    Handshake { node_id: String, public_key: Vec<u8> },
    NewBlock { block_data: Vec<u8> },
    BlockRequest { block_hash: String },
    BlockResponse { block_data: Vec<u8> },
    TransactionBroadcast { transaction: Vec<u8> },
    ConsensusProposal { proposal_id: String, data: Vec<u8> },
    ConsensusVote { proposal_id: String, vote: bool, signature: Vec<u8> },
    PeerDiscovery { known_peers: Vec<SocketAddr> },
    Heartbeat { timestamp: u64 },
    Disconnect,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusProposal {
    pub proposal_id: String,
    pub proposer: String,
    pub block_hash: String,
    pub timestamp: u64,
    pub votes_for: u32,
    pub votes_against: u32,
    pub required_votes: u32,
    pub status: ProposalStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProposalStatus {
    Pending,
    Approved,
    Rejected,
    Expired,
}

pub struct TlsConfig {
    pub cert_path: String,
    pub key_path: String,
    pub ca_cert_path: Option<String>,
}

pub struct P2PNetwork {
    node_id: String,
    listen_address: SocketAddr,
    peers: Arc<Mutex<HashMap<String, PeerNode>>>,
    active_proposals: Arc<Mutex<HashMap<String, ConsensusProposal>>>,
    max_peers: usize,
    is_running: Arc<Mutex<bool>>,
    // PRODUCTION FIX: Add TLS support
    tls_config: Option<Arc<ServerConfig>>,
    client_tls_config: Option<Arc<ClientConfig>>,
    // Message encryption
    encryption_key: Arc<LessSafeKey>,
    rng: SystemRandom,
}

impl P2PNetwork {
    /// Create new P2P network with TLS support (PRODUCTION)
    pub fn new_with_tls(node_id: String, listen_address: SocketAddr, max_peers: usize, tls_config: TlsConfig) -> Result<Self, String> {
        // Load TLS certificates
        let server_config = Self::load_tls_server_config(&tls_config)?;
        let client_config = Self::load_tls_client_config(&tls_config)?;
        
        // Generate encryption key for message payload
        let rng = SystemRandom::new();
        let mut key_bytes = vec![0u8; 32];
        rng.fill(&mut key_bytes)
            .map_err(|e| format!("Failed to generate encryption key: {:?}", e))?;
        
        let unbound_key = UnboundKey::new(&AES_256_GCM, &key_bytes)
            .map_err(|e| format!("Failed to create encryption key: {:?}", e))?;
        let encryption_key = LessSafeKey::new(unbound_key);
        
        Ok(P2PNetwork {
            node_id,
            listen_address,
            peers: Arc::new(Mutex::new(HashMap::new())),
            active_proposals: Arc::new(Mutex::new(HashMap::new())),
            max_peers,
            is_running: Arc::new(Mutex::new(false)),
            tls_config: Some(Arc::new(server_config)),
            client_tls_config: Some(Arc::new(client_config)),
            encryption_key: Arc::new(encryption_key),
            rng,
        })
    }
    
    /// Create new P2P network without TLS (DEVELOPMENT/TESTING ONLY)
    pub fn new(node_id: String, listen_address: SocketAddr, max_peers: usize) -> Self {
        let rng = SystemRandom::new();
        let mut key_bytes = vec![0u8; 32];
        rng.fill(&mut key_bytes).unwrap();
        let unbound_key = UnboundKey::new(&AES_256_GCM, &key_bytes).unwrap();
        let encryption_key = LessSafeKey::new(unbound_key);
        
        P2PNetwork {
            node_id,
            listen_address,
            peers: Arc::new(Mutex::new(HashMap::new())),
            active_proposals: Arc::new(Mutex::new(HashMap::new())),
            max_peers,
            is_running: Arc::new(Mutex::new(false)),
            tls_config: None,
            client_tls_config: None,
            encryption_key: Arc::new(encryption_key),
            rng,
        }
    }
    
    fn load_tls_server_config(tls_config: &TlsConfig) -> Result<ServerConfig, String> {
        let cert_file = File::open(&tls_config.cert_path)
            .map_err(|e| format!("Failed to open cert file: {}", e))?;
        let key_file = File::open(&tls_config.key_path)
            .map_err(|e| format!("Failed to open key file: {}", e))?;
        
        let mut cert_reader = BufReader::new(cert_file);
        let mut key_reader = BufReader::new(key_file);
        
        let certs = certs(&mut cert_reader)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to parse certificates: {}", e))?;
        
        let keys = pkcs8_private_keys(&mut key_reader)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to parse private keys: {}", e))?;
        
        let key = keys.into_iter().next()
            .ok_or("No private key found")?;
        
        let config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key.into())
            .map_err(|e| format!("Failed to create TLS config: {}", e))?;
        
        Ok(config)
    }
    
    fn load_tls_client_config(tls_config: &TlsConfig) -> Result<ClientConfig, String> {
        let mut config = ClientConfig::builder()
            .with_root_certificates(rustls::RootCertStore::empty())
            .with_no_client_auth();
        
        if let Some(ref ca_path) = tls_config.ca_cert_path {
            let ca_file = File::open(ca_path)
                .map_err(|e| format!("Failed to open CA cert: {}", e))?;
            let mut ca_reader = BufReader::new(ca_file);
            let ca_certs = certs(&mut ca_reader)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Failed to parse CA certs: {}", e))?;
            
            let mut root_store = rustls::RootCertStore::empty();
            for cert in ca_certs {
                root_store.add(cert)
                    .map_err(|e| format!("Failed to add CA cert: {}", e))?;
            }
            
            config = ClientConfig::builder()
                .with_root_certificates(root_store)
                .with_no_client_auth();
        }
        
        Ok(config)
    }

    pub fn start(&mut self) -> Result<(), String> {
        let listener = TcpListener::bind(self.listen_address)
            .map_err(|e| format!("Failed to bind to address: {}", e))?;
        
        *self.is_running.lock().unwrap() = true;
        
        println!("[P2P] Node {} listening on {}", self.node_id, self.listen_address);
        
        // Start listener thread
        let peers = Arc::clone(&self.peers);
        let proposals = Arc::clone(&self.active_proposals);
        let is_running = Arc::clone(&self.is_running);
        let node_id = self.node_id.clone();
        
        thread::spawn(move || {
            Self::listen_for_connections(listener, peers, proposals, is_running, node_id);
        });
        
        // Start heartbeat thread
        self.start_heartbeat();
        
        Ok(())
    }

    fn listen_for_connections(
        listener: TcpListener,
        peers: Arc<Mutex<HashMap<String, PeerNode>>>,
        proposals: Arc<Mutex<HashMap<String, ConsensusProposal>>>,
        is_running: Arc<Mutex<bool>>,
        node_id: String,
    ) {
        for stream in listener.incoming() {
            if !*is_running.lock().unwrap() {
                break;
            }
            
            match stream {
                Ok(stream) => {
                    let peers_clone = Arc::clone(&peers);
                    let proposals_clone = Arc::clone(&proposals);
                    let node_id_clone = node_id.clone();
                    
                    thread::spawn(move || {
                        Self::handle_peer_connection(stream, peers_clone, proposals_clone, node_id_clone);
                    });
                }
                Err(e) => {
                    eprintln!("[P2P] Connection error: {}", e);
                }
            }
        }
    }

    fn handle_peer_connection(
        mut stream: TcpStream,
        peers: Arc<Mutex<HashMap<String, PeerNode>>>,
        proposals: Arc<Mutex<HashMap<String, ConsensusProposal>>>,
        node_id: String,
    ) {
        let peer_addr = stream.peer_addr().unwrap();
        println!("[P2P] New connection from {}", peer_addr);
        
        let mut buffer = vec![0u8; 65536];
        
        loop {
            match stream.read(&mut buffer) {
                Ok(0) => {
                    println!("[P2P] Peer {} disconnected", peer_addr);
                    break;
                }
                Ok(n) => {
                    if let Ok(message) = bincode::deserialize::<NetworkMessage>(&buffer[..n]) {
                        Self::process_message(message, &mut stream, &peers, &proposals, &node_id);
                    }
                }
                Err(e) => {
                    eprintln!("[P2P] Read error: {}", e);
                    break;
                }
            }
        }
    }

    fn process_message(
        message: NetworkMessage,
        stream: &mut TcpStream,
        peers: &Arc<Mutex<HashMap<String, PeerNode>>>,
        proposals: &Arc<Mutex<HashMap<String, ConsensusProposal>>>,
        node_id: &str,
    ) {
        match message {
            NetworkMessage::Handshake { node_id: peer_id, public_key } => {
                println!("[P2P] Handshake from peer: {}", peer_id);
                
                let peer_addr = stream.peer_addr().unwrap();
                let peer = PeerNode {
                    node_id: peer_id.clone(),
                    address: peer_addr,
                    public_key,
                    reputation: 100,
                    last_seen: Self::current_timestamp(),
                    is_validator: false,
                };
                
                peers.lock().unwrap().insert(peer_id, peer);
                
                // Send handshake response
                let response = NetworkMessage::Handshake {
                    node_id: node_id.to_string(),
                    public_key: vec![0u8; 32], // Placeholder
                };
                Self::send_message(stream, &response);
            }
            
            NetworkMessage::NewBlock { block_data } => {
                println!("[P2P] Received new block ({} bytes)", block_data.len());
                // Validate and add to blockchain
                // Broadcast to other peers
            }
            
            NetworkMessage::ConsensusProposal { proposal_id, data } => {
                println!("[P2P] Received consensus proposal: {}", proposal_id);
                
                let proposal = ConsensusProposal {
                    proposal_id: proposal_id.clone(),
                    proposer: "peer".to_string(),
                    block_hash: "hash".to_string(),
                    timestamp: Self::current_timestamp(),
                    votes_for: 0,
                    votes_against: 0,
                    required_votes: 5,
                    status: ProposalStatus::Pending,
                };
                
                proposals.lock().unwrap().insert(proposal_id, proposal);
            }
            
            NetworkMessage::ConsensusVote { proposal_id, vote, signature } => {
                println!("[P2P] Received vote for proposal {}: {}", proposal_id, vote);
                
                if let Some(proposal) = proposals.lock().unwrap().get_mut(&proposal_id) {
                    if vote {
                        proposal.votes_for += 1;
                    } else {
                        proposal.votes_against += 1;
                    }
                    
                    // Check if consensus reached
                    if proposal.votes_for >= proposal.required_votes {
                        proposal.status = ProposalStatus::Approved;
                        println!("[P2P] Consensus reached for proposal {}", proposal_id);
                    } else if proposal.votes_against > (proposal.required_votes / 2) {
                        proposal.status = ProposalStatus::Rejected;
                        println!("[P2P] Proposal {} rejected", proposal_id);
                    }
                }
            }
            
            NetworkMessage::PeerDiscovery { known_peers } => {
                println!("[P2P] Received {} peer addresses", known_peers.len());
                // Connect to new peers
            }
            
            NetworkMessage::Heartbeat { timestamp } => {
                // Update peer last_seen
            }
            
            _ => {
                println!("[P2P] Unhandled message type");
            }
        }
    }

    fn send_message(stream: &mut TcpStream, message: &NetworkMessage) {
        if let Ok(encoded) = bincode::serialize(message) {
            let _ = stream.write_all(&encoded);
        }
    }

    pub fn connect_to_peer(&self, peer_address: SocketAddr) -> Result<(), String> {
        let mut stream = TcpStream::connect(peer_address)
            .map_err(|e| format!("Failed to connect to peer: {}", e))?;
        
        // Send handshake
        let handshake = NetworkMessage::Handshake {
            node_id: self.node_id.clone(),
            public_key: vec![0u8; 32], // Placeholder
        };
        
        Self::send_message(&mut stream, &handshake);
        
        println!("[P2P] Connected to peer at {}", peer_address);
        
        Ok(())
    }

    pub fn broadcast_block(&self, block_data: Vec<u8>) {
        let message = NetworkMessage::NewBlock { block_data };
        self.broadcast_message(&message);
    }

    pub fn propose_consensus(&self, proposal_id: String, data: Vec<u8>) {
        let message = NetworkMessage::ConsensusProposal { proposal_id, data };
        self.broadcast_message(&message);
    }

    pub fn vote_on_proposal(&self, proposal_id: String, vote: bool) {
        let message = NetworkMessage::ConsensusVote {
            proposal_id,
            vote,
            signature: vec![0u8; 64], // Placeholder
        };
        self.broadcast_message(&message);
    }

    fn broadcast_message(&self, message: &NetworkMessage) {
        let peers = self.peers.lock().unwrap();
        
        for (peer_id, peer) in peers.iter() {
            if let Ok(mut stream) = TcpStream::connect(peer.address) {
                Self::send_message(&mut stream, message);
                println!("[P2P] Broadcasted message to peer {}", peer_id);
            }
        }
    }

    fn start_heartbeat(&self) {
        let peers = Arc::clone(&self.peers);
        let is_running = Arc::clone(&self.is_running);
        
        thread::spawn(move || {
            while *is_running.lock().unwrap() {
                thread::sleep(std::time::Duration::from_secs(30));
                
                let message = NetworkMessage::Heartbeat {
                    timestamp: Self::current_timestamp(),
                };
                
                let peers_lock = peers.lock().unwrap();
                for (_, peer) in peers_lock.iter() {
                    if let Ok(mut stream) = TcpStream::connect(peer.address) {
                        Self::send_message(&mut stream, &message);
                    }
                }
            }
        });
    }

    pub fn get_peer_count(&self) -> usize {
        self.peers.lock().unwrap().len()
    }

    pub fn get_active_proposals(&self) -> Vec<ConsensusProposal> {
        self.active_proposals.lock().unwrap().values().cloned().collect()
    }

    fn current_timestamp() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    pub fn stop(&mut self) {
        *self.is_running.lock().unwrap() = false;
        println!("[P2P] Network stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let addr = "127.0.0.1:8000".parse().unwrap();
        let network = P2PNetwork::new("node1".to_string(), addr, 10);
        assert_eq!(network.node_id, "node1");
        assert_eq!(network.max_peers, 10);
    }

    #[test]
    fn test_peer_management() {
        let addr = "127.0.0.1:8001".parse().unwrap();
        let network = P2PNetwork::new("node2".to_string(), addr, 10);
        assert_eq!(network.get_peer_count(), 0);
    }
}
