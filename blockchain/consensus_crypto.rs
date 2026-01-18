// Consensus Cryptographic Verification
// Production-ready vote signature verification and BFT guarantees

use ring::signature::{Ed25519KeyPair, UnparsedPublicKey, ED25519};
use serde::{Serialize, Deserialize};
use sha3::{Sha3_256, Digest};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedVote {
    pub block_hash: String,
    pub validator_id: String,
    pub vote: bool,
    pub timestamp: i64,
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BftProof {
    pub block_hash: String,
    pub votes: Vec<SignedVote>,
    pub total_validators: usize,
    pub positive_votes: usize,
    pub consensus_threshold: f64,
    pub proof_timestamp: i64,
}

pub struct ConsensusVerifier {
    validator_public_keys: HashMap<String, Vec<u8>>,
}

impl ConsensusVerifier {
    pub fn new() -> Self {
        ConsensusVerifier {
            validator_public_keys: HashMap::new(),
        }
    }

    /// Register validator public key
    pub fn register_validator(&mut self, validator_id: String, public_key: Vec<u8>) -> Result<(), String> {
        if public_key.len() != 32 {
            return Err("Invalid public key length".to_string());
        }
        self.validator_public_keys.insert(validator_id, public_key);
        Ok(())
    }

    /// Verify vote signature
    pub fn verify_vote_signature(&self, vote: &SignedVote) -> Result<bool, String> {
        // Get validator's public key
        let public_key = self.validator_public_keys.get(&vote.validator_id)
            .ok_or_else(|| format!("Validator {} not registered", vote.validator_id))?;

        // Reconstruct signed message
        let message = self.construct_vote_message(&vote.block_hash, &vote.validator_id, vote.vote, vote.timestamp);

        // Verify signature
        let public_key_verifier = UnparsedPublicKey::new(&ED25519, public_key);
        match public_key_verifier.verify(&message, &vote.signature) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Construct vote message for signing/verification
    fn construct_vote_message(&self, block_hash: &str, validator_id: &str, vote: bool, timestamp: i64) -> Vec<u8> {
        let mut message = Vec::new();
        message.extend_from_slice(block_hash.as_bytes());
        message.extend_from_slice(validator_id.as_bytes());
        message.push(if vote { 1 } else { 0 });
        message.extend_from_slice(&timestamp.to_le_bytes());
        message
    }

    /// Generate BFT proof for committed block
    pub fn generate_bft_proof(
        &self,
        block_hash: &str,
        votes: Vec<SignedVote>,
        total_validators: usize,
        consensus_threshold: f64,
    ) -> Result<BftProof, String> {
        // Verify all vote signatures
        for vote in &votes {
            if !self.verify_vote_signature(vote)? {
                return Err(format!("Invalid signature from validator {}", vote.validator_id));
            }
        }

        let positive_votes = votes.iter().filter(|v| v.vote).count();
        let vote_ratio = positive_votes as f64 / total_validators as f64;

        if vote_ratio < consensus_threshold {
            return Err(format!(
                "Insufficient votes: {}/{} ({:.2}% < {:.2}%)",
                positive_votes, total_validators,
                vote_ratio * 100.0, consensus_threshold * 100.0
            ));
        }

        Ok(BftProof {
            block_hash: block_hash.to_string(),
            votes,
            total_validators,
            positive_votes,
            consensus_threshold,
            proof_timestamp: chrono::Utc::now().timestamp(),
        })
    }

    /// Verify BFT proof
    pub fn verify_bft_proof(&self, proof: &BftProof) -> Result<bool, String> {
        // Verify all signatures
        for vote in &proof.votes {
            if vote.block_hash != proof.block_hash {
                return Err("Vote block hash mismatch".to_string());
            }
            if !self.verify_vote_signature(vote)? {
                return Err(format!("Invalid signature from validator {}", vote.validator_id));
            }
        }

        // Verify consensus threshold
        let vote_ratio = proof.positive_votes as f64 / proof.total_validators as f64;
        if vote_ratio < proof.consensus_threshold {
            return Err("Consensus threshold not met".to_string());
        }

        Ok(true)
    }

    /// Check for Byzantine behavior (conflicting votes)
    pub fn detect_byzantine_behavior(&self, votes: &[SignedVote]) -> Vec<String> {
        let mut validator_votes: HashMap<String, Vec<&SignedVote>> = HashMap::new();
        
        for vote in votes {
            validator_votes.entry(vote.validator_id.clone())
                .or_insert_with(Vec::new)
                .push(vote);
        }

        let mut byzantine_validators = Vec::new();
        for (validator_id, validator_vote_list) in validator_votes {
            // Check for conflicting votes (same block, different vote values)
            let mut block_votes: HashMap<String, Vec<bool>> = HashMap::new();
            for vote in validator_vote_list {
                block_votes.entry(vote.block_hash.clone())
                    .or_insert_with(Vec::new)
                    .push(vote.vote);
            }

            for (_, votes_for_block) in block_votes {
                if votes_for_block.len() > 1 {
                    let has_conflict = votes_for_block.iter().any(|&v| v != votes_for_block[0]);
                    if has_conflict {
                        byzantine_validators.push(validator_id.clone());
                        break;
                    }
                }
            }
        }

        byzantine_validators
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ring::rand::SystemRandom;
    use ring::signature::KeyPair;

    #[test]
    fn test_vote_signature_verification() {
        let rng = SystemRandom::new();
        let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref()).unwrap();
        let public_key = key_pair.public_key().as_ref().to_vec();

        let mut verifier = ConsensusVerifier::new();
        verifier.register_validator("validator1".to_string(), public_key.clone()).unwrap();

        let block_hash = "test_block_hash";
        let validator_id = "validator1";
        let vote = true;
        let timestamp = chrono::Utc::now().timestamp();

        let message = verifier.construct_vote_message(block_hash, validator_id, vote, timestamp);
        let signature = key_pair.sign(&message).as_ref().to_vec();

        let signed_vote = SignedVote {
            block_hash: block_hash.to_string(),
            validator_id: validator_id.to_string(),
            vote,
            timestamp,
            signature,
            public_key,
        };

        assert!(verifier.verify_vote_signature(&signed_vote).unwrap());
    }

    #[test]
    fn test_bft_proof_generation() {
        let rng = SystemRandom::new();
        let mut verifier = ConsensusVerifier::new();
        let mut votes = Vec::new();

        // Create 4 validators
        for i in 0..4 {
            let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
            let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref()).unwrap();
            let public_key = key_pair.public_key().as_ref().to_vec();
            let validator_id = format!("validator{}", i);

            verifier.register_validator(validator_id.clone(), public_key.clone()).unwrap();

            let block_hash = "test_block";
            let timestamp = chrono::Utc::now().timestamp();
            let message = verifier.construct_vote_message(block_hash, &validator_id, true, timestamp);
            let signature = key_pair.sign(&message).as_ref().to_vec();

            votes.push(SignedVote {
                block_hash: block_hash.to_string(),
                validator_id,
                vote: true,
                timestamp,
                signature,
                public_key,
            });
        }

        let proof = verifier.generate_bft_proof("test_block", votes, 4, 0.67).unwrap();
        assert_eq!(proof.positive_votes, 4);
        assert!(verifier.verify_bft_proof(&proof).unwrap());
    }

    #[test]
    fn test_byzantine_detection() {
        let rng = SystemRandom::new();
        let mut verifier = ConsensusVerifier::new();
        
        let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref()).unwrap();
        let public_key = key_pair.public_key().as_ref().to_vec();
        
        verifier.register_validator("byzantine_validator".to_string(), public_key.clone()).unwrap();

        let block_hash = "test_block";
        let validator_id = "byzantine_validator";
        let timestamp = chrono::Utc::now().timestamp();

        // Create conflicting votes
        let mut votes = Vec::new();
        for vote_value in [true, false] {
            let message = verifier.construct_vote_message(block_hash, validator_id, vote_value, timestamp);
            let signature = key_pair.sign(&message).as_ref().to_vec();
            votes.push(SignedVote {
                block_hash: block_hash.to_string(),
                validator_id: validator_id.to_string(),
                vote: vote_value,
                timestamp,
                signature: signature.clone(),
                public_key: public_key.clone(),
            });
        }

        let byzantine = verifier.detect_byzantine_behavior(&votes);
        assert_eq!(byzantine.len(), 1);
        assert_eq!(byzantine[0], "byzantine_validator");
    }
}
