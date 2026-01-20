// Comprehensive tests for Blockchain Consensus Mechanism
// Tests: PoA consensus, vote verification, BFT, validator management

use super::*;
use std::sync::Arc;
use std::collections::HashMap;

#[cfg(test)]
mod consensus_tests {
    use super::*;

    fn setup_consensus() -> ConsensusEngine {
        ConsensusEngine::new()
    }

    fn create_test_validator(id: &str, stake: u64) -> Validator {
        use ring::signature::{Ed25519KeyPair, KeyPair};
        use ring::rand::SystemRandom;
        
        let rng = SystemRandom::new();
        let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref()).unwrap();
        
        Validator {
            id: id.to_string(),
            public_key: key_pair.public_key().as_ref().to_vec(),
            stake,
            reputation: 100,
            is_active: true,
            total_votes: 0,
            successful_proposals: 0,
        }
    }

    #[test]
    fn test_consensus_initialization() {
        let consensus = setup_consensus();
        assert_eq!(consensus.get_validator_count(), 0);
        assert_eq!(consensus.get_consensus_threshold(), 67); // 67% for BFT
    }

    #[test]
    fn test_register_validator() {
        let mut consensus = setup_consensus();
        let validator = create_test_validator("validator1", 1000);
        
        let result = consensus.register_validator(validator.clone());
        assert!(result.is_ok());
        assert_eq!(consensus.get_validator_count(), 1);
        
        let retrieved = consensus.get_validator(&validator.id).unwrap();
        assert_eq!(retrieved.id, validator.id);
        assert_eq!(retrieved.stake, validator.stake);
    }

    #[test]
    fn test_duplicate_validator_rejection() {
        let mut consensus = setup_consensus();
        let validator = create_test_validator("validator1", 1000);
        
        consensus.register_validator(validator.clone()).unwrap();
        let result = consensus.register_validator(validator);
        
        assert!(result.is_err(), "Duplicate validator should be rejected");
    }

    #[test]
    fn test_minimum_stake_requirement() {
        let mut consensus = setup_consensus();
        let validator = create_test_validator("validator1", 100); // Below minimum
        
        let result = consensus.register_validator(validator);
        assert!(result.is_err(), "Validator with insufficient stake should be rejected");
    }

    #[test]
    fn test_propose_block() {
        let mut consensus = setup_consensus();
        
        // Register validators
        for i in 0..3 {
            let validator = create_test_validator(&format!("validator{}", i), 1000);
            consensus.register_validator(validator).unwrap();
        }
        
        let block_hash = "test_block_hash_123";
        let proposer_id = "validator0";
        
        let result = consensus.propose_block(proposer_id, block_hash);
        assert!(result.is_ok());
    }

    #[test]
    fn test_vote_block_with_signature_verification() {
        let mut consensus = setup_consensus();
        
        // Register validators with keypairs
        let mut validators_with_keys = vec![];
        for i in 0..3 {
            let (validator, key_pair) = create_validator_with_keypair(&format!("validator{}", i), 1000);
            consensus.register_validator(validator.clone()).unwrap();
            validators_with_keys.push((validator, key_pair));
        }
        
        // Propose block
        let block_hash = "test_block_hash_123";
        consensus.propose_block("validator0", block_hash).unwrap();
        
        // Vote with valid signature
        let (validator, key_pair) = &validators_with_keys[1];
        let vote = create_signed_vote(block_hash, &validator.id, true, key_pair);
        
        let result = consensus.vote_block(vote);
        assert!(result.is_ok(), "Valid vote signature should be accepted");
    }

    #[test]
    fn test_invalid_vote_signature_rejection() {
        let mut consensus = setup_consensus();
        
        let (validator, _) = create_validator_with_keypair("validator1", 1000);
        consensus.register_validator(validator.clone()).unwrap();
        
        let block_hash = "test_block_hash_123";
        consensus.propose_block(&validator.id, block_hash).unwrap();
        
        // Create vote with invalid signature
        let mut vote = BlockVote {
            block_hash: block_hash.to_string(),
            validator_id: validator.id.clone(),
            vote: true,
            timestamp: chrono::Utc::now().timestamp(),
            signature: vec![0u8; 64], // Invalid signature
        };
        
        let result = consensus.vote_block(vote);
        assert!(result.is_err(), "Invalid vote signature should be rejected");
    }

    #[test]
    fn test_consensus_threshold_67_percent() {
        let mut consensus = setup_consensus();
        
        // Register 3 validators
        let mut validators_with_keys = vec![];
        for i in 0..3 {
            let (validator, key_pair) = create_validator_with_keypair(&format!("validator{}", i), 1000);
            consensus.register_validator(validator.clone()).unwrap();
            validators_with_keys.push((validator, key_pair));
        }
        
        let block_hash = "test_block_hash_123";
        consensus.propose_block("validator0", block_hash).unwrap();
        
        // 2 out of 3 votes (67%) should reach consensus
        for i in 0..2 {
            let (validator, key_pair) = &validators_with_keys[i];
            let vote = create_signed_vote(block_hash, &validator.id, true, key_pair);
            consensus.vote_block(vote).unwrap();
        }
        
        let has_consensus = consensus.check_consensus(block_hash).unwrap();
        assert!(has_consensus, "67% votes should reach consensus");
    }

    #[test]
    fn test_consensus_not_reached_below_threshold() {
        let mut consensus = setup_consensus();
        
        // Register 3 validators
        let mut validators_with_keys = vec![];
        for i in 0..3 {
            let (validator, key_pair) = create_validator_with_keypair(&format!("validator{}", i), 1000);
            consensus.register_validator(validator.clone()).unwrap();
            validators_with_keys.push((validator, key_pair));
        }
        
        let block_hash = "test_block_hash_123";
        consensus.propose_block("validator0", block_hash).unwrap();
        
        // Only 1 out of 3 votes (33%) - below threshold
        let (validator, key_pair) = &validators_with_keys[0];
        let vote = create_signed_vote(block_hash, &validator.id, true, key_pair);
        consensus.vote_block(vote).unwrap();
        
        let has_consensus = consensus.check_consensus(block_hash).unwrap();
        assert!(!has_consensus, "33% votes should not reach consensus");
    }

    #[test]
    fn test_byzantine_fault_tolerance() {
        let mut consensus = setup_consensus();
        
        // Register 4 validators (can tolerate 1 Byzantine)
        let mut validators_with_keys = vec![];
        for i in 0..4 {
            let (validator, key_pair) = create_validator_with_keypair(&format!("validator{}", i), 1000);
            consensus.register_validator(validator.clone()).unwrap();
            validators_with_keys.push((validator, key_pair));
        }
        
        let block_hash = "test_block_hash_123";
        consensus.propose_block("validator0", block_hash).unwrap();
        
        // 3 honest votes (75%), 1 Byzantine (will vote against)
        for i in 0..3 {
            let (validator, key_pair) = &validators_with_keys[i];
            let vote = create_signed_vote(block_hash, &validator.id, true, key_pair);
            consensus.vote_block(vote).unwrap();
        }
        
        // Byzantine validator votes against
        let (validator, key_pair) = &validators_with_keys[3];
        let vote = create_signed_vote(block_hash, &validator.id, false, key_pair);
        consensus.vote_block(vote).unwrap();
        
        let has_consensus = consensus.check_consensus(block_hash).unwrap();
        assert!(has_consensus, "Should reach consensus despite 1 Byzantine validator");
    }

    #[test]
    fn test_validator_reputation_update() {
        let mut consensus = setup_consensus();
        
        let (validator, key_pair) = create_validator_with_keypair("validator1", 1000);
        let initial_reputation = validator.reputation;
        consensus.register_validator(validator.clone()).unwrap();
        
        let block_hash = "test_block_hash_123";
        consensus.propose_block(&validator.id, block_hash).unwrap();
        
        // Successful vote increases reputation
        let vote = create_signed_vote(block_hash, &validator.id, true, &key_pair);
        consensus.vote_block(vote).unwrap();
        
        let updated_validator = consensus.get_validator(&validator.id).unwrap();
        assert!(updated_validator.reputation >= initial_reputation);
    }

    #[test]
    fn test_inactive_validator_cannot_vote() {
        let mut consensus = setup_consensus();
        
        let (mut validator, key_pair) = create_validator_with_keypair("validator1", 1000);
        validator.is_active = false;
        consensus.register_validator(validator.clone()).unwrap();
        
        let block_hash = "test_block_hash_123";
        consensus.propose_block(&validator.id, block_hash).unwrap();
        
        let vote = create_signed_vote(block_hash, &validator.id, true, &key_pair);
        let result = consensus.vote_block(vote);
        
        assert!(result.is_err(), "Inactive validator should not be able to vote");
    }

    #[test]
    fn test_double_voting_prevention() {
        let mut consensus = setup_consensus();
        
        let (validator, key_pair) = create_validator_with_keypair("validator1", 1000);
        consensus.register_validator(validator.clone()).unwrap();
        
        let block_hash = "test_block_hash_123";
        consensus.propose_block(&validator.id, block_hash).unwrap();
        
        // First vote
        let vote1 = create_signed_vote(block_hash, &validator.id, true, &key_pair);
        consensus.vote_block(vote1).unwrap();
        
        // Second vote (should be rejected)
        let vote2 = create_signed_vote(block_hash, &validator.id, true, &key_pair);
        let result = consensus.vote_block(vote2);
        
        assert!(result.is_err(), "Double voting should be prevented");
    }

    #[test]
    fn test_proposal_timeout() {
        let mut consensus = setup_consensus();
        
        let (validator, _) = create_validator_with_keypair("validator1", 1000);
        consensus.register_validator(validator.clone()).unwrap();
        
        let block_hash = "test_block_hash_123";
        consensus.propose_block(&validator.id, block_hash).unwrap();
        
        // Simulate time passing (proposal timeout)
        std::thread::sleep(std::time::Duration::from_secs(consensus.get_proposal_timeout() + 1));
        
        let is_expired = consensus.is_proposal_expired(block_hash).unwrap();
        assert!(is_expired, "Proposal should expire after timeout");
    }

    #[test]
    fn test_concurrent_proposals() {
        let mut consensus = setup_consensus();
        
        // Register validators
        for i in 0..3 {
            let (validator, _) = create_validator_with_keypair(&format!("validator{}", i), 1000);
            consensus.register_validator(validator).unwrap();
        }
        
        // Multiple proposals
        let result1 = consensus.propose_block("validator0", "block_hash_1");
        let result2 = consensus.propose_block("validator1", "block_hash_2");
        let result3 = consensus.propose_block("validator2", "block_hash_3");
        
        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_ok());
        
        assert_eq!(consensus.get_pending_proposals_count(), 3);
    }

    #[test]
    fn test_validator_slashing_for_malicious_behavior() {
        let mut consensus = setup_consensus();
        
        let (validator, key_pair) = create_validator_with_keypair("validator1", 1000);
        let initial_stake = validator.stake;
        consensus.register_validator(validator.clone()).unwrap();
        
        // Simulate malicious behavior (e.g., signing conflicting blocks)
        consensus.slash_validator(&validator.id, 10).unwrap(); // 10% slash
        
        let updated_validator = consensus.get_validator(&validator.id).unwrap();
        assert!(updated_validator.stake < initial_stake, "Stake should be reduced after slashing");
    }

    // Helper functions
    fn create_validator_with_keypair(id: &str, stake: u64) -> (Validator, Ed25519KeyPair) {
        use ring::signature::{Ed25519KeyPair, KeyPair};
        use ring::rand::SystemRandom;
        
        let rng = SystemRandom::new();
        let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref()).unwrap();
        
        let validator = Validator {
            id: id.to_string(),
            public_key: key_pair.public_key().as_ref().to_vec(),
            stake,
            reputation: 100,
            is_active: true,
            total_votes: 0,
            successful_proposals: 0,
        };
        
        (validator, key_pair)
    }

    fn create_signed_vote(
        block_hash: &str,
        validator_id: &str,
        vote: bool,
        key_pair: &Ed25519KeyPair
    ) -> BlockVote {
        let timestamp = chrono::Utc::now().timestamp();
        
        let mut message = Vec::new();
        message.extend_from_slice(block_hash.as_bytes());
        message.extend_from_slice(validator_id.as_bytes());
        message.push(if vote { 1 } else { 0 });
        message.extend_from_slice(&timestamp.to_le_bytes());
        
        let signature = key_pair.sign(&message);
        
        BlockVote {
            block_hash: block_hash.to_string(),
            validator_id: validator_id.to_string(),
            vote,
            timestamp,
            signature: signature.as_ref().to_vec(),
        }
    }
}
