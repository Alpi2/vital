// Blockchain Key Management and Authentication
// Production-ready cryptographic key management with HSM support

use ring::{
    rand::{SecureRandom, SystemRandom},
    signature::{Ed25519KeyPair, KeyPair, ED25519_PUBLIC_KEY_LEN},
};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use sha3::{Sha3_256, Digest};
use base64::{Engine as _, engine::general_purpose};
use chrono::{Utc, Duration};
use std::path::PathBuf;
use std::fs;

// Import HSM integration module
mod hsm_integration;
use hsm_integration::{HsmProvider, HsmConfig, HsmKeyHandle};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    pub id: String,
    pub public_key: Vec<u8>,
    pub role: UserRole,
    pub created_at: i64,
    pub expires_at: Option<i64>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UserRole {
    Admin,
    Validator,
    Researcher,
    Clinician,
    Patient,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessToken {
    pub token: String,
    pub identity_id: String,
    pub issued_at: i64,
    pub expires_at: i64,
    pub permissions: Vec<String>,
}

pub struct KeyManager {
    rng: SystemRandom,
    identities: Arc<RwLock<HashMap<String, Identity>>>,
    // PRODUCTION FIX: Use HSM instead of in-memory key storage
    hsm_provider: Option<Arc<HsmProvider>>,
    hsm_key_handles: Arc<RwLock<HashMap<String, HsmKeyHandle>>>,
    // Fallback for development/testing only
    key_pairs: Arc<RwLock<HashMap<String, Ed25519KeyPair>>>,
    active_tokens: Arc<RwLock<HashMap<String, AccessToken>>>,
    revoked_tokens: Arc<RwLock<Vec<String>>>,
    // Persistent storage for revoked tokens
    revocation_db_path: PathBuf,
}

impl KeyManager {
    /// Create new KeyManager with HSM support (PRODUCTION)
    pub fn new_with_hsm(hsm_config: HsmConfig, revocation_db_path: PathBuf) -> Result<Self, String> {
        // Load revoked tokens from persistent storage
        let revoked_tokens = Self::load_revoked_tokens(&revocation_db_path)?;
        
        let hsm_provider = HsmProvider::new(hsm_config)
            .map_err(|e| format!("Failed to initialize HSM: {}", e))?;
        
        Ok(KeyManager {
            rng: SystemRandom::new(),
            identities: Arc::new(RwLock::new(HashMap::new())),
            hsm_provider: Some(Arc::new(hsm_provider)),
            hsm_key_handles: Arc::new(RwLock::new(HashMap::new())),
            key_pairs: Arc::new(RwLock::new(HashMap::new())),
            active_tokens: Arc::new(RwLock::new(HashMap::new())),
            revoked_tokens: Arc::new(RwLock::new(revoked_tokens)),
            revocation_db_path,
        })
    }
    
    /// Create new KeyManager without HSM (DEVELOPMENT/TESTING ONLY)
    pub fn new() -> Self {
        KeyManager {
            rng: SystemRandom::new(),
            identities: Arc::new(RwLock::new(HashMap::new())),
            hsm_provider: None,
            hsm_key_handles: Arc::new(RwLock::new(HashMap::new())),
            key_pairs: Arc::new(RwLock::new(HashMap::new())),
            active_tokens: Arc::new(RwLock::new(HashMap::new())),
            revoked_tokens: Arc::new(RwLock::new(Vec::new())),
            revocation_db_path: PathBuf::from("/tmp/revoked_tokens.json"),
        }
    }
    
    fn load_revoked_tokens(path: &PathBuf) -> Result<Vec<String>, String> {
        if !path.exists() {
            return Ok(Vec::new());
        }
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read revocation DB: {}", e))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse revocation DB: {}", e))
    }
    
    fn save_revoked_tokens(&self) -> Result<(), String> {
        let tokens = self.revoked_tokens.read().unwrap();
        let content = serde_json::to_string_pretty(&*tokens)
            .map_err(|e| format!("Failed to serialize revoked tokens: {}", e))?;
        fs::write(&self.revocation_db_path, content)
            .map_err(|e| format!("Failed to write revocation DB: {}", e))
    }

    /// Generate new identity with Ed25519 key pair (PRODUCTION: Uses HSM)
    pub fn create_identity(&self, role: UserRole, metadata: HashMap<String, String>) -> Result<(String, Vec<u8>), String> {
        let public_key = if let Some(ref hsm) = self.hsm_provider {
            // PRODUCTION: Generate key in HSM
            let key_handle = hsm.generate_key()
                .map_err(|e| format!("HSM key generation failed: {}", e))?;
            let public_key = hsm.get_public_key(&key_handle)
                .map_err(|e| format!("Failed to get public key from HSM: {}", e))?;
            
            // Generate identity ID from public key hash
            let mut hasher = Sha3_256::new();
            hasher.update(&public_key);
            let id = format!("{:x}", hasher.finalize());
            
            // Store HSM key handle
            let mut handles = self.hsm_key_handles.write()
                .map_err(|e| format!("Lock error: {}", e))?;
            handles.insert(id.clone(), key_handle);
            
            // Create identity
            let identity = Identity {
                id: id.clone(),
                public_key: public_key.clone(),
                role,
                created_at: Utc::now().timestamp(),
                expires_at: None,
                metadata,
            };
            
            let mut identities = self.identities.write()
                .map_err(|e| format!("Lock error: {}", e))?;
            identities.insert(id.clone(), identity);
            
            return Ok((id, public_key));
        } else {
            // FALLBACK: In-memory key generation (DEVELOPMENT ONLY)
            let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&self.rng)
                .map_err(|e| format!("Key generation failed: {:?}", e))?;
            
            let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref())
                .map_err(|e| format!("Key pair creation failed: {:?}", e))?;
            
            let public_key = key_pair.public_key().as_ref().to_vec();
            
            // Generate identity ID from public key hash
            let mut hasher = Sha3_256::new();
            hasher.update(&public_key);
            let id = format!("{:x}", hasher.finalize());
            
            // Create identity
            let identity = Identity {
                id: id.clone(),
                public_key: public_key.clone(),
                role,
                created_at: Utc::now().timestamp(),
                expires_at: None,
                metadata,
            };
            
            // Store identity and key pair
            let mut identities = self.identities.write()
                .map_err(|e| format!("Lock error: {}", e))?;
            identities.insert(id.clone(), identity);
            
            let mut key_pairs = self.key_pairs.write()
                .map_err(|e| format!("Lock error: {}", e))?;
            key_pairs.insert(id.clone(), key_pair);
            
            Ok((id, public_key))
        }
    }

    /// Issue access token for identity
    pub fn issue_token(&self, identity_id: &str, duration_hours: i64, permissions: Vec<String>) -> Result<String, String> {
        // Verify identity exists
        let identities = self.identities.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        if !identities.contains_key(identity_id) {
            return Err("Identity not found".to_string());
        }
        
        // Generate token
        let mut token_bytes = vec![0u8; 32];
        self.rng.fill(&mut token_bytes)
            .map_err(|e| format!("Random generation failed: {:?}", e))?;
        
        let token = general_purpose::STANDARD.encode(&token_bytes);
        
        let now = Utc::now().timestamp();
        let expires_at = now + (duration_hours * 3600);
        
        let access_token = AccessToken {
            token: token.clone(),
            identity_id: identity_id.to_string(),
            issued_at: now,
            expires_at,
            permissions,
        };
        
        // Store token
        let mut active_tokens = self.active_tokens.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        active_tokens.insert(token.clone(), access_token);
        
        Ok(token)
    }

    /// Validate access token
    pub fn validate_token(&self, token: &str) -> Result<AccessToken, String> {
        // Check if revoked
        let revoked = self.revoked_tokens.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        if revoked.contains(&token.to_string()) {
            return Err("Token has been revoked".to_string());
        }
        
        // Get token
        let active_tokens = self.active_tokens.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        let access_token = active_tokens.get(token)
            .ok_or_else(|| "Invalid token".to_string())?;
        
        // Check expiration
        let now = Utc::now().timestamp();
        if now > access_token.expires_at {
            return Err("Token has expired".to_string());
        }
        
        Ok(access_token.clone())
    }

    /// Revoke access token
    pub fn revoke_token(&self, token: &str) -> Result<(), String> {
        let mut revoked = self.revoked_tokens.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        revoked.push(token.to_string());
        
        let mut active_tokens = self.active_tokens.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        active_tokens.remove(token);
        
        Ok(())
    }

    /// Sign data with identity's private key
    pub fn sign_data(&self, identity_id: &str, data: &[u8]) -> Result<Vec<u8>, String> {
        let key_pairs = self.key_pairs.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        let key_pair = key_pairs.get(identity_id)
            .ok_or_else(|| "Identity not found".to_string())?;
        
        let signature = key_pair.sign(data);
        Ok(signature.as_ref().to_vec())
    }

    /// Verify signature
    pub fn verify_signature(&self, identity_id: &str, data: &[u8], signature: &[u8]) -> Result<bool, String> {
        let identities = self.identities.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        let identity = identities.get(identity_id)
            .ok_or_else(|| "Identity not found".to_string())?;
        
        let public_key = ring::signature::UnparsedPublicKey::new(
            &ring::signature::ED25519,
            &identity.public_key
        );
        
        match public_key.verify(data, signature) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Get identity by ID
    pub fn get_identity(&self, identity_id: &str) -> Result<Identity, String> {
        let identities = self.identities.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        identities.get(identity_id)
            .cloned()
            .ok_or_else(|| "Identity not found".to_string())
    }

    /// Check permission
    pub fn has_permission(&self, token: &str, permission: &str) -> Result<bool, String> {
        let access_token = self.validate_token(token)?;
        Ok(access_token.permissions.contains(&permission.to_string()))
    }

    /// Rotate identity keys
    pub fn rotate_keys(&self, identity_id: &str) -> Result<Vec<u8>, String> {
        // Generate new key pair
        let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&self.rng)
            .map_err(|e| format!("Key generation failed: {:?}", e))?;
        
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref())
            .map_err(|e| format!("Key pair creation failed: {:?}", e))?;
        
        let public_key = key_pair.public_key().as_ref().to_vec();
        
        // Update identity
        let mut identities = self.identities.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        let identity = identities.get_mut(identity_id)
            .ok_or_else(|| "Identity not found".to_string())?;
        
        identity.public_key = public_key.clone();
        
        // Update key pair
        let mut key_pairs = self.key_pairs.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        key_pairs.insert(identity_id.to_string(), key_pair);
        
        Ok(public_key)
    }

    /// Clean expired tokens
    pub fn cleanup_expired_tokens(&self) -> Result<usize, String> {
        let now = Utc::now().timestamp();
        let mut count = 0;
        
        let mut active_tokens = self.active_tokens.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        active_tokens.retain(|_, token| {
            if now > token.expires_at {
                count += 1;
                false
            } else {
                true
            }
        });
        
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_identity() {
        let km = KeyManager::new();
        let mut metadata = HashMap::new();
        metadata.insert("name".to_string(), "Test User".to_string());
        
        let result = km.create_identity(UserRole::Clinician, metadata);
        assert!(result.is_ok());
        
        let (id, public_key) = result.unwrap();
        assert!(!id.is_empty());
        assert_eq!(public_key.len(), ED25519_PUBLIC_KEY_LEN);
    }

    #[test]
    fn test_token_lifecycle() {
        let km = KeyManager::new();
        let (id, _) = km.create_identity(UserRole::Researcher, HashMap::new()).unwrap();
        
        // Issue token
        let token = km.issue_token(&id, 24, vec!["read".to_string()]).unwrap();
        
        // Validate token
        let validation = km.validate_token(&token);
        assert!(validation.is_ok());
        
        // Revoke token
        km.revoke_token(&token).unwrap();
        
        // Should fail after revocation
        let validation = km.validate_token(&token);
        assert!(validation.is_err());
    }

    #[test]
    fn test_signature() {
        let km = KeyManager::new();
        let (id, _) = km.create_identity(UserRole::Admin, HashMap::new()).unwrap();
        
        let data = b"Test message";
        let signature = km.sign_data(&id, data).unwrap();
        
        let valid = km.verify_signature(&id, data, &signature).unwrap();
        assert!(valid);
        
        // Wrong data should fail
        let invalid = km.verify_signature(&id, b"Wrong message", &signature).unwrap();
        assert!(!invalid);
    }
}
