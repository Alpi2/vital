// Quantum-Resistant Cryptography Implementation
// NIST Post-Quantum Cryptography (PQC) Standards
// Using Kyber-512 (KEM) and Dilithium (Digital Signatures)

use pqcrypto_kyber::kyber512::*;
use pqcrypto_dilithium::dilithium2::*;
use pqcrypto_traits::kem::{PublicKey, SecretKey, SharedSecret, Ciphertext};
use pqcrypto_traits::sign::{PublicKey as SignPublicKey, SecretKey as SignSecretKey, SignedMessage, DetachedSignature};
use sha3::{Sha3_256, Digest};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use aes_gcm::aead::{Aead, NewAead};
use rand::rngs::OsRng;
use serde::{Serialize, Deserialize};
use std::error::Error;
use std::fmt;

/// Quantum-resistant key pair for key encapsulation
#[derive(Clone)]
pub struct QuantumKeyPair {
    pub public_key: Vec<u8>,
    pub secret_key: Vec<u8>,
}

/// Quantum-resistant signature key pair
#[derive(Clone)]
pub struct QuantumSignatureKeyPair {
    pub public_key: Vec<u8>,
    pub secret_key: Vec<u8>,
}

/// Encrypted message with quantum-resistant encryption
#[derive(Serialize, Deserialize, Clone)]
pub struct QuantumEncryptedMessage {
    pub ciphertext: Vec<u8>,
    pub encapsulated_key: Vec<u8>,
    pub nonce: Vec<u8>,
    pub signature: Vec<u8>,
}

/// Quantum crypto error types
#[derive(Debug)]
pub enum QuantumCryptoError {
    KeyGenerationError,
    EncryptionError,
    DecryptionError,
    SignatureError,
    VerificationError,
    InvalidKeyLength,
}

impl fmt::Display for QuantumCryptoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            QuantumCryptoError::KeyGenerationError => write!(f, "Failed to generate quantum-resistant keys"),
            QuantumCryptoError::EncryptionError => write!(f, "Encryption failed"),
            QuantumCryptoError::DecryptionError => write!(f, "Decryption failed"),
            QuantumCryptoError::SignatureError => write!(f, "Signature generation failed"),
            QuantumCryptoError::VerificationError => write!(f, "Signature verification failed"),
            QuantumCryptoError::InvalidKeyLength => write!(f, "Invalid key length"),
        }
    }
}

impl Error for QuantumCryptoError {}

/// Quantum-resistant cryptography manager
pub struct QuantumCryptoManager {
    kem_keypair: Option<QuantumKeyPair>,
    signature_keypair: Option<QuantumSignatureKeyPair>,
}

impl QuantumCryptoManager {
    /// Create a new quantum crypto manager
    pub fn new() -> Self {
        QuantumCryptoManager {
            kem_keypair: None,
            signature_keypair: None,
        }
    }

    /// Generate Kyber-512 key pair for key encapsulation
    pub fn generate_kem_keypair(&mut self) -> Result<(), QuantumCryptoError> {
        let (pk, sk) = keypair();
        
        self.kem_keypair = Some(QuantumKeyPair {
            public_key: pk.as_bytes().to_vec(),
            secret_key: sk.as_bytes().to_vec(),
        });
        
        Ok(())
    }

    /// Generate Dilithium-2 key pair for digital signatures
    pub fn generate_signature_keypair(&mut self) -> Result<(), QuantumCryptoError> {
        let (pk, sk) = pqcrypto_dilithium::dilithium2::keypair();
        
        self.signature_keypair = Some(QuantumSignatureKeyPair {
            public_key: pk.as_bytes().to_vec(),
            secret_key: sk.as_bytes().to_vec(),
        });
        
        Ok(())
    }

    /// Encrypt data using quantum-resistant hybrid encryption
    /// Uses Kyber-512 for key encapsulation + AES-256-GCM for data encryption
    pub fn encrypt(
        &self,
        plaintext: &[u8],
        recipient_public_key: &[u8],
    ) -> Result<QuantumEncryptedMessage, QuantumCryptoError> {
        // Reconstruct recipient's public key
        let pk = PublicKey::from_bytes(recipient_public_key)
            .map_err(|_| QuantumCryptoError::InvalidKeyLength)?;
        
        // Encapsulate a shared secret using Kyber-512
        let (ss, ct) = encapsulate(&pk);
        
        // Derive AES key from shared secret
        let mut hasher = Sha3_256::new();
        hasher.update(ss.as_bytes());
        let aes_key = hasher.finalize();
        
        // Encrypt plaintext with AES-256-GCM
        let cipher = Aes256Gcm::new(Key::from_slice(&aes_key));
        let nonce_bytes = rand::random::<[u8; 12]>();
        let nonce = Nonce::from_slice(&nonce_bytes);
        
        let ciphertext = cipher.encrypt(nonce, plaintext)
            .map_err(|_| QuantumCryptoError::EncryptionError)?;
        
        // Sign the encrypted message
        let signature = self.sign_message(&ciphertext)?;
        
        Ok(QuantumEncryptedMessage {
            ciphertext,
            encapsulated_key: ct.as_bytes().to_vec(),
            nonce: nonce_bytes.to_vec(),
            signature,
        })
    }

    /// Decrypt quantum-resistant encrypted message
    pub fn decrypt(
        &self,
        encrypted_msg: &QuantumEncryptedMessage,
    ) -> Result<Vec<u8>, QuantumCryptoError> {
        let keypair = self.kem_keypair.as_ref()
            .ok_or(QuantumCryptoError::DecryptionError)?;
        
        // Reconstruct secret key and ciphertext
        let sk = SecretKey::from_bytes(&keypair.secret_key)
            .map_err(|_| QuantumCryptoError::InvalidKeyLength)?;
        let ct = Ciphertext::from_bytes(&encrypted_msg.encapsulated_key)
            .map_err(|_| QuantumCryptoError::InvalidKeyLength)?;
        
        // Decapsulate shared secret
        let ss = decapsulate(&ct, &sk);
        
        // Derive AES key
        let mut hasher = Sha3_256::new();
        hasher.update(ss.as_bytes());
        let aes_key = hasher.finalize();
        
        // Decrypt with AES-256-GCM
        let cipher = Aes256Gcm::new(Key::from_slice(&aes_key));
        let nonce = Nonce::from_slice(&encrypted_msg.nonce);
        
        let plaintext = cipher.decrypt(nonce, encrypted_msg.ciphertext.as_ref())
            .map_err(|_| QuantumCryptoError::DecryptionError)?;
        
        Ok(plaintext)
    }

    /// Sign a message using Dilithium-2
    pub fn sign_message(&self, message: &[u8]) -> Result<Vec<u8>, QuantumCryptoError> {
        let keypair = self.signature_keypair.as_ref()
            .ok_or(QuantumCryptoError::SignatureError)?;
        
        let sk = SignSecretKey::from_bytes(&keypair.secret_key)
            .map_err(|_| QuantumCryptoError::InvalidKeyLength)?;
        
        let signed_msg = pqcrypto_dilithium::dilithium2::sign(message, &sk);
        Ok(signed_msg.as_bytes().to_vec())
    }

    /// Verify a signature using Dilithium-2
    pub fn verify_signature(
        &self,
        message: &[u8],
        signature: &[u8],
        public_key: &[u8],
    ) -> Result<bool, QuantumCryptoError> {
        let pk = SignPublicKey::from_bytes(public_key)
            .map_err(|_| QuantumCryptoError::InvalidKeyLength)?;
        
        let signed_msg = SignedMessage::from_bytes(signature)
            .map_err(|_| QuantumCryptoError::VerificationError)?;
        
        match pqcrypto_dilithium::dilithium2::open(&signed_msg, &pk) {
            Ok(verified_msg) => Ok(verified_msg == message),
            Err(_) => Ok(false),
        }
    }

    /// Get public key for key encapsulation
    pub fn get_kem_public_key(&self) -> Option<Vec<u8>> {
        self.kem_keypair.as_ref().map(|kp| kp.public_key.clone())
    }

    /// Get public key for signatures
    pub fn get_signature_public_key(&self) -> Option<Vec<u8>> {
        self.signature_keypair.as_ref().map(|kp| kp.public_key.clone())
    }
}

/// Hybrid encryption scheme combining classical and quantum-resistant algorithms
pub struct HybridEncryption {
    quantum_manager: QuantumCryptoManager,
}

impl HybridEncryption {
    pub fn new() -> Self {
        let mut manager = QuantumCryptoManager::new();
        manager.generate_kem_keypair().expect("Failed to generate KEM keypair");
        manager.generate_signature_keypair().expect("Failed to generate signature keypair");
        
        HybridEncryption {
            quantum_manager: manager,
        }
    }

    /// Encrypt medical data with quantum-resistant encryption
    pub fn encrypt_medical_data(
        &self,
        patient_data: &[u8],
        recipient_public_key: &[u8],
    ) -> Result<QuantumEncryptedMessage, QuantumCryptoError> {
        self.quantum_manager.encrypt(patient_data, recipient_public_key)
    }

    /// Decrypt medical data
    pub fn decrypt_medical_data(
        &self,
        encrypted_data: &QuantumEncryptedMessage,
    ) -> Result<Vec<u8>, QuantumCryptoError> {
        self.quantum_manager.decrypt(encrypted_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_generation() {
        let mut manager = QuantumCryptoManager::new();
        assert!(manager.generate_kem_keypair().is_ok());
        assert!(manager.generate_signature_keypair().is_ok());
        assert!(manager.get_kem_public_key().is_some());
        assert!(manager.get_signature_public_key().is_some());
    }

    #[test]
    fn test_encryption_decryption() {
        let mut sender = QuantumCryptoManager::new();
        sender.generate_kem_keypair().unwrap();
        sender.generate_signature_keypair().unwrap();
        
        let mut receiver = QuantumCryptoManager::new();
        receiver.generate_kem_keypair().unwrap();
        
        let plaintext = b"Sensitive medical data: Patient ID 12345";
        let recipient_pk = receiver.get_kem_public_key().unwrap();
        
        let encrypted = sender.encrypt(plaintext, &recipient_pk).unwrap();
        let decrypted = receiver.decrypt(&encrypted).unwrap();
        
        assert_eq!(plaintext.to_vec(), decrypted);
    }

    #[test]
    fn test_signature_verification() {
        let mut manager = QuantumCryptoManager::new();
        manager.generate_signature_keypair().unwrap();
        
        let message = b"Medical record hash: abc123";
        let signature = manager.sign_message(message).unwrap();
        let public_key = manager.get_signature_public_key().unwrap();
        
        let is_valid = manager.verify_signature(message, &signature, &public_key).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_hybrid_encryption() {
        let sender = HybridEncryption::new();
        let receiver = HybridEncryption::new();
        
        let patient_data = b"Patient: John Doe, Diagnosis: Hypertension";
        let recipient_pk = receiver.quantum_manager.get_kem_public_key().unwrap();
        
        let encrypted = sender.encrypt_medical_data(patient_data, &recipient_pk).unwrap();
        let decrypted = receiver.decrypt_medical_data(&encrypted).unwrap();
        
        assert_eq!(patient_data.to_vec(), decrypted);
    }
}
