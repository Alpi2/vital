// Comprehensive tests for Blockchain Ledger Storage
// Tests: RocksDB persistence, transaction validation, signature verification, backup/restore

use super::*;
use tempfile::TempDir;
use std::sync::Arc;

#[cfg(test)]
mod ledger_storage_tests {
    use super::*;

    fn setup_test_ledger() -> (LedgerStorage, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let ledger = LedgerStorage::new(temp_dir.path().to_str().unwrap()).unwrap();
        (ledger, temp_dir)
    }

    #[test]
    fn test_ledger_initialization() {
        let (ledger, _temp) = setup_test_ledger();
        assert!(ledger.get_latest_block().is_ok());
        
        // Genesis block should exist
        let genesis = ledger.get_latest_block().unwrap();
        assert_eq!(genesis.index, 0);
        assert_eq!(genesis.previous_hash, "0");
    }

    #[test]
    fn test_add_block_success() {
        let (mut ledger, _temp) = setup_test_ledger();
        
        let transactions = vec![
            create_test_transaction("tx1", "alice", "bob", "data1"),
            create_test_transaction("tx2", "bob", "charlie", "data2"),
        ];
        
        let result = ledger.add_block(transactions.clone());
        assert!(result.is_ok());
        
        let block = result.unwrap();
        assert_eq!(block.index, 1);
        assert_eq!(block.transactions.len(), 2);
    }

    #[test]
    fn test_transaction_signature_verification() {
        let (mut ledger, _temp) = setup_test_ledger();
        
        // Create transaction with valid signature
        let (public_key, private_key) = generate_ed25519_keypair();
        let transaction = create_signed_transaction(
            "tx1",
            &public_key,
            "recipient",
            "test_data",
            &private_key
        );
        
        let result = ledger.add_block(vec![transaction]);
        assert!(result.is_ok(), "Valid signature should be accepted");
    }

    #[test]
    fn test_invalid_signature_rejection() {
        let (mut ledger, _temp) = setup_test_ledger();
        
        // Create transaction with invalid signature
        let mut transaction = create_test_transaction("tx1", "alice", "bob", "data");
        transaction.signature = vec![0u8; 64]; // Invalid signature
        
        let result = ledger.add_block(vec![transaction]);
        assert!(result.is_err(), "Invalid signature should be rejected");
    }

    #[test]
    fn test_block_hash_validation() {
        let (mut ledger, _temp) = setup_test_ledger();
        
        let transactions = vec![create_test_transaction("tx1", "alice", "bob", "data")];
        let block = ledger.add_block(transactions).unwrap();
        
        // Verify hash is correctly calculated
        let calculated_hash = ledger.calculate_block_hash(&block);
        assert_eq!(block.hash, calculated_hash);
    }

    #[test]
    fn test_block_chain_integrity() {
        let (mut ledger, _temp) = setup_test_ledger();
        
        // Add multiple blocks
        for i in 0..10 {
            let tx = create_test_transaction(
                &format!("tx{}", i),
                "alice",
                "bob",
                &format!("data{}", i)
            );
            ledger.add_block(vec![tx]).unwrap();
        }
        
        // Verify chain integrity
        let latest = ledger.get_latest_block().unwrap();
        assert_eq!(latest.index, 10);
        
        // Verify each block links to previous
        for i in 1..=10 {
            let block = ledger.get_block_by_index(i).unwrap();
            let prev_block = ledger.get_block_by_index(i - 1).unwrap();
            assert_eq!(block.previous_hash, prev_block.hash);
        }
    }

    #[test]
    fn test_persistence_across_restarts() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().to_str().unwrap();
        
        // Create ledger and add blocks
        {
            let mut ledger = LedgerStorage::new(db_path).unwrap();
            for i in 0..5 {
                let tx = create_test_transaction(
                    &format!("tx{}", i),
                    "alice",
                    "bob",
                    &format!("data{}", i)
                );
                ledger.add_block(vec![tx]).unwrap();
            }
        }
        
        // Reopen ledger and verify data persisted
        {
            let ledger = LedgerStorage::new(db_path).unwrap();
            let latest = ledger.get_latest_block().unwrap();
            assert_eq!(latest.index, 5);
        }
    }

    #[test]
    fn test_backup_and_restore() {
        let (mut ledger, _temp) = setup_test_ledger();
        let backup_dir = TempDir::new().unwrap();
        
        // Add some blocks
        for i in 0..5 {
            let tx = create_test_transaction(
                &format!("tx{}", i),
                "alice",
                "bob",
                &format!("data{}", i)
            );
            ledger.add_block(vec![tx]).unwrap();
        }
        
        // Backup
        let backup_path = backup_dir.path().join("backup.db");
        ledger.backup(backup_path.to_str().unwrap()).unwrap();
        
        // Restore to new ledger
        let restore_dir = TempDir::new().unwrap();
        let mut new_ledger = LedgerStorage::new(restore_dir.path().to_str().unwrap()).unwrap();
        new_ledger.restore(backup_path.to_str().unwrap()).unwrap();
        
        // Verify data matches
        let original_latest = ledger.get_latest_block().unwrap();
        let restored_latest = new_ledger.get_latest_block().unwrap();
        assert_eq!(original_latest.index, restored_latest.index);
        assert_eq!(original_latest.hash, restored_latest.hash);
    }

    #[test]
    fn test_concurrent_block_additions() {
        use std::thread;
        
        let (ledger, _temp) = setup_test_ledger();
        let ledger = Arc::new(Mutex::new(ledger));
        
        let mut handles = vec![];
        
        // Spawn 10 threads adding blocks concurrently
        for i in 0..10 {
            let ledger_clone = Arc::clone(&ledger);
            let handle = thread::spawn(move || {
                let tx = create_test_transaction(
                    &format!("tx{}", i),
                    "alice",
                    "bob",
                    &format!("data{}", i)
                );
                let mut ledger = ledger_clone.lock().unwrap();
                ledger.add_block(vec![tx])
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            assert!(handle.join().unwrap().is_ok());
        }
        
        // Verify all blocks added
        let ledger = ledger.lock().unwrap();
        let latest = ledger.get_latest_block().unwrap();
        assert_eq!(latest.index, 10);
    }

    #[test]
    fn test_transaction_validation_empty_fields() {
        let (mut ledger, _temp) = setup_test_ledger();
        
        // Empty transaction ID
        let mut tx = create_test_transaction("", "alice", "bob", "data");
        assert!(ledger.add_block(vec![tx]).is_err());
        
        // Empty sender
        tx = create_test_transaction("tx1", "", "bob", "data");
        assert!(ledger.add_block(vec![tx]).is_err());
        
        // Empty signature
        tx = create_test_transaction("tx1", "alice", "bob", "data");
        tx.signature = vec![];
        assert!(ledger.add_block(vec![tx]).is_err());
    }

    #[test]
    fn test_get_block_by_hash() {
        let (mut ledger, _temp) = setup_test_ledger();
        
        let tx = create_test_transaction("tx1", "alice", "bob", "data");
        let block = ledger.add_block(vec![tx]).unwrap();
        let block_hash = block.hash.clone();
        
        let retrieved = ledger.get_block_by_hash(&block_hash).unwrap();
        assert_eq!(retrieved.hash, block_hash);
        assert_eq!(retrieved.index, block.index);
    }

    #[test]
    fn test_database_compaction() {
        let (mut ledger, _temp) = setup_test_ledger();
        
        // Add many blocks
        for i in 0..100 {
            let tx = create_test_transaction(
                &format!("tx{}", i),
                "alice",
                "bob",
                &format!("data{}", i)
            );
            ledger.add_block(vec![tx]).unwrap();
        }
        
        // Compact database
        let result = ledger.compact_database();
        assert!(result.is_ok());
        
        // Verify data still accessible
        let latest = ledger.get_latest_block().unwrap();
        assert_eq!(latest.index, 100);
    }

    #[test]
    fn test_transaction_timestamp_validation() {
        let (mut ledger, _temp) = setup_test_ledger();
        
        let mut tx = create_test_transaction("tx1", "alice", "bob", "data");
        
        // Future timestamp should be rejected
        tx.timestamp = chrono::Utc::now().timestamp() + 3600;
        assert!(ledger.add_block(vec![tx.clone()]).is_err());
        
        // Very old timestamp should be rejected
        tx.timestamp = chrono::Utc::now().timestamp() - 86400 * 365;
        assert!(ledger.add_block(vec![tx]).is_err());
    }

    #[test]
    fn test_block_size_limits() {
        let (mut ledger, _temp) = setup_test_ledger();
        
        // Create block with too many transactions
        let mut transactions = vec![];
        for i in 0..10000 {
            transactions.push(create_test_transaction(
                &format!("tx{}", i),
                "alice",
                "bob",
                "data"
            ));
        }
        
        let result = ledger.add_block(transactions);
        assert!(result.is_err(), "Block size limit should be enforced");
    }

    // Helper functions
    fn create_test_transaction(id: &str, from: &str, to: &str, data: &str) -> Transaction {
        use ring::signature::{Ed25519KeyPair, KeyPair};
        use ring::rand::SystemRandom;
        
        let rng = SystemRandom::new();
        let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref()).unwrap();
        
        let public_key = base64::encode(key_pair.public_key().as_ref());
        
        let mut message = Vec::new();
        message.extend_from_slice(id.as_bytes());
        message.extend_from_slice(public_key.as_bytes());
        message.extend_from_slice(to.as_bytes());
        message.extend_from_slice("patient_data".as_bytes());
        message.extend_from_slice(data.as_bytes());
        message.extend_from_slice(&chrono::Utc::now().timestamp().to_le_bytes());
        
        let signature = key_pair.sign(&message);
        
        Transaction {
            id: id.to_string(),
            from: public_key,
            to: to.to_string(),
            data_type: "patient_data".to_string(),
            data_hash: data.to_string(),
            timestamp: chrono::Utc::now().timestamp(),
            signature: signature.as_ref().to_vec(),
        }
    }

    fn generate_ed25519_keypair() -> (String, Ed25519KeyPair) {
        use ring::signature::{Ed25519KeyPair, KeyPair};
        use ring::rand::SystemRandom;
        
        let rng = SystemRandom::new();
        let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref()).unwrap();
        let public_key = base64::encode(key_pair.public_key().as_ref());
        
        (public_key, key_pair)
    }

    fn create_signed_transaction(
        id: &str,
        public_key: &str,
        to: &str,
        data: &str,
        key_pair: &Ed25519KeyPair
    ) -> Transaction {
        let timestamp = chrono::Utc::now().timestamp();
        
        let mut message = Vec::new();
        message.extend_from_slice(id.as_bytes());
        message.extend_from_slice(public_key.as_bytes());
        message.extend_from_slice(to.as_bytes());
        message.extend_from_slice("patient_data".as_bytes());
        message.extend_from_slice(data.as_bytes());
        message.extend_from_slice(&timestamp.to_le_bytes());
        
        let signature = key_pair.sign(&message);
        
        Transaction {
            id: id.to_string(),
            from: public_key.to_string(),
            to: to.to_string(),
            data_type: "patient_data".to_string(),
            data_hash: data.to_string(),
            timestamp,
            signature: signature.as_ref().to_vec(),
        }
    }
}
