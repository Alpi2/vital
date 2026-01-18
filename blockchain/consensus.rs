// Distributed Consensus Mechanism
// Production-ready Proof-of-Authority (PoA) consensus with Byzantine Fault Tolerance

use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use sha3::{Sha3_256, Digest};
use chrono::Utc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsensusMessage {
    ProposeBlock(BlockProposal),
    VoteBlock(BlockVote),
    CommitBlock(BlockCommit),
    ValidatorHeartbeat(ValidatorStatus),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BlockProposal {
    pub block_hash: String,
    pub block_index: u64,
    pub proposer_id: String,
    pub timestamp: i64,
    pub transactions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BlockVote {
    pub block_hash: String,
    pub validator_id: String,
    pub vote: bool,
    pub timestamp: i64,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BlockCommit {
    pub block_hash: String,
    pub votes: Vec<BlockVote>,
    pub committed_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorStatus {
    pub validator_id: String,
    pub is_active: bool,
    pub last_seen: i64,
    pub blocks_validated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Validator {
    pub id: String,
    pub public_key: Vec<u8>,
    pub stake: u64,
    pub reputation: f64,
    pub is_active: bool,
    pub joined_at: i64,
}

pub struct ConsensusEngine {
    validators: Arc<RwLock<HashMap<String, Validator>>>,
    pending_proposals: Arc<RwLock<HashMap<String, BlockProposal>>>,
    votes: Arc<RwLock<HashMap<String, Vec<BlockVote>>>>,
    committed_blocks: Arc<RwLock<HashSet<String>>>,
    min_validators: usize,
    consensus_threshold: f64,
}

impl ConsensusEngine {
    pub fn new(min_validators: usize, consensus_threshold: f64) -> Self {
        ConsensusEngine {
            validators: Arc::new(RwLock::new(HashMap::new())),
            pending_proposals: Arc::new(RwLock::new(HashMap::new())),
            votes: Arc::new(RwLock::new(HashMap::new())),
            committed_blocks: Arc::new(RwLock::new(HashSet::new())),
            min_validators,
            consensus_threshold,
        }
    }

    /// Register new validator
    pub fn register_validator(&self, validator: Validator) -> Result<(), String> {
        let mut validators = self.validators.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        if validators.contains_key(&validator.id) {
            return Err("Validator already registered".to_string());
        }
        
        validators.insert(validator.id.clone(), validator);
        Ok(())
    }

    /// Propose new block
    pub fn propose_block(&self, proposal: BlockProposal) -> Result<(), String> {
        // Verify proposer is active validator
        let validators = self.validators.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        let validator = validators.get(&proposal.proposer_id)
            .ok_or_else(|| "Proposer is not a registered validator".to_string())?;
        
        if !validator.is_active {
            return Err("Proposer is not active".to_string());
        }
        
        // Check if block already proposed
        let mut proposals = self.pending_proposals.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        if proposals.contains_key(&proposal.block_hash) {
            return Err("Block already proposed".to_string());
        }
        
        proposals.insert(proposal.block_hash.clone(), proposal);
        
        // Initialize vote tracking
        let mut votes = self.votes.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        votes.insert(proposal.block_hash.clone(), Vec::new());
        
        Ok(())
    }

    /// Submit vote for block with signature verification
    pub fn vote_block(&self, vote: BlockVote) -> Result<(), String> {
        // Verify voter is active validator
        let validators = self.validators.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        let validator = validators.get(&vote.validator_id)
            .ok_or_else(|| "Voter is not a registered validator".to_string())?;
        
        if !validator.is_active {
            return Err("Voter is not active".to_string());
        }
        
        // CRITICAL: Verify vote signature using Ed25519
        use ring::signature::{UnparsedPublicKey, ED25519};
        
        // Reconstruct the message that was signed
        let mut message = Vec::new();
        message.extend_from_slice(vote.block_hash.as_bytes());
        message.extend_from_slice(vote.validator_id.as_bytes());
        message.push(if vote.vote { 1 } else { 0 });
        message.extend_from_slice(&vote.timestamp.to_le_bytes());
        
        // Verify signature with validator's public key
        if validator.public_key.len() != 32 {
            return Err("Invalid validator public key length".to_string());
        }
        
        let public_key = UnparsedPublicKey::new(&ED25519, &validator.public_key);
        public_key.verify(&message, &vote.signature)
            .map_err(|_| "Invalid vote signature - cryptographic verification failed".to_string())?;
        
        // Verify block proposal exists
        let proposals = self.pending_proposals.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        if !proposals.contains_key(&vote.block_hash) {
            return Err("Block proposal not found".to_string());
        }
        
        // Add vote
        let mut votes = self.votes.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        let block_votes = votes.get_mut(&vote.block_hash)
            .ok_or_else(|| "Vote tracking not initialized".to_string())?;
        
        // Check if validator already voted
        if block_votes.iter().any(|v| v.validator_id == vote.validator_id) {
            return Err("Validator already voted for this block".to_string());
        }
        
        block_votes.push(vote);
        
        Ok(())
    }

    /// Check if block has reached consensus
    pub fn check_consensus(&self, block_hash: &str) -> Result<bool, String> {
        let validators = self.validators.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        let active_validators: Vec<_> = validators.values()
            .filter(|v| v.is_active)
            .collect();
        
        if active_validators.len() < self.min_validators {
            return Ok(false);
        }
        
        let votes = self.votes.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        let block_votes = votes.get(block_hash)
            .ok_or_else(|| "Block not found".to_string())?;
        
        // Count positive votes
        let positive_votes = block_votes.iter()
            .filter(|v| v.vote)
            .count();
        
        let vote_ratio = positive_votes as f64 / active_validators.len() as f64;
        
        Ok(vote_ratio >= self.consensus_threshold)
    }

    /// Commit block after consensus
    pub fn commit_block(&self, block_hash: &str) -> Result<BlockCommit, String> {
        // Verify consensus reached
        if !self.check_consensus(block_hash)? {
            return Err("Consensus not reached".to_string());
        }
        
        let votes = self.votes.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        let block_votes = votes.get(block_hash)
            .ok_or_else(|| "Block not found".to_string())?
            .clone();
        
        let commit = BlockCommit {
            block_hash: block_hash.to_string(),
            votes: block_votes,
            committed_at: Utc::now().timestamp(),
        };
        
        // Mark as committed
        let mut committed = self.committed_blocks.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        committed.insert(block_hash.to_string());
        
        // Clean up
        let mut proposals = self.pending_proposals.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        proposals.remove(block_hash);
        
        Ok(commit)
    }

    /// Get active validators
    pub fn get_active_validators(&self) -> Result<Vec<Validator>, String> {
        let validators = self.validators.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        Ok(validators.values()
            .filter(|v| v.is_active)
            .cloned()
            .collect())
    }

    /// Update validator status
    pub fn update_validator_status(&self, validator_id: &str, is_active: bool) -> Result<(), String> {
        let mut validators = self.validators.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        let validator = validators.get_mut(validator_id)
            .ok_or_else(|| "Validator not found".to_string())?;
        
        validator.is_active = is_active;
        Ok(())
    }

    /// Update validator reputation
    pub fn update_reputation(&self, validator_id: &str, delta: f64) -> Result<(), String> {
        let mut validators = self.validators.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        let validator = validators.get_mut(validator_id)
            .ok_or_else(|| "Validator not found".to_string())?;
        
        validator.reputation = (validator.reputation + delta).max(0.0).min(100.0);
        Ok(())
    }

    /// Get consensus statistics
    pub fn get_stats(&self) -> Result<ConsensusStats, String> {
        let validators = self.validators.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        let proposals = self.pending_proposals.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        let committed = self.committed_blocks.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        Ok(ConsensusStats {
            total_validators: validators.len(),
            active_validators: validators.values().filter(|v| v.is_active).count(),
            pending_proposals: proposals.len(),
            committed_blocks: committed.len(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConsensusStats {
    pub total_validators: usize,
    pub active_validators: usize,
    pub pending_proposals: usize,
    pub committed_blocks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_registration() {
        let engine = ConsensusEngine::new(3, 0.67);
        
        let validator = Validator {
            id: "validator1".to_string(),
            public_key: vec![1, 2, 3],
            stake: 1000,
            reputation: 100.0,
            is_active: true,
            joined_at: Utc::now().timestamp(),
        };
        
        assert!(engine.register_validator(validator).is_ok());
    }

    #[test]
    fn test_consensus_flow() {
        let engine = ConsensusEngine::new(3, 0.67);
        
        // Register validators
        for i in 1..=3 {
            let validator = Validator {
                id: format!("validator{}", i),
                public_key: vec![i as u8],
                stake: 1000,
                reputation: 100.0,
                is_active: true,
                joined_at: Utc::now().timestamp(),
            };
            engine.register_validator(validator).unwrap();
        }
        
        // Propose block
        let proposal = BlockProposal {
            block_hash: "test_block".to_string(),
            block_index: 1,
            proposer_id: "validator1".to_string(),
            timestamp: Utc::now().timestamp(),
            transactions: vec![],
        };
        
        engine.propose_block(proposal).unwrap();
        
        // Vote
        for i in 1..=3 {
            let vote = BlockVote {
                block_hash: "test_block".to_string(),
                validator_id: format!("validator{}", i),
                vote: true,
                timestamp: Utc::now().timestamp(),
                signature: vec![],
            };
            engine.vote_block(vote).unwrap();
        }
        
        // Check consensus
        assert!(engine.check_consensus("test_block").unwrap());
        
        // Commit
        let commit = engine.commit_block("test_block");
        assert!(commit.is_ok());
    }
}
