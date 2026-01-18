// Persistent Blockchain Ledger Storage
// Production-ready distributed ledger with RocksDB backend

use rocksdb::{DB, Options, WriteBatch, IteratorMode};
use serde::{Serialize, Deserialize};
use std::path::Path;
use std::sync::{Arc, RwLock};
use sha3::{Sha3_256, Digest};
use chrono::Utc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub index: u64,
    pub timestamp: i64,
    pub data: Vec<u8>,
    pub previous_hash: String,
    pub hash: String,
    pub nonce: u64,
    pub validator: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub id: String,
    pub from: String,
    pub to: String,
    pub data_type: String,
    pub data_hash: String,
    pub timestamp: i64,
    pub signature: Vec<u8>,
}

pub struct LedgerStorage {
    db: Arc<DB>,
    chain_lock: Arc<RwLock<Vec<Block>>>,
    pending_transactions: Arc<RwLock<Vec<Transaction>>>,
}

impl LedgerStorage {
    /// Initialize persistent ledger storage
    pub fn new(db_path: &str) -> Result<Self, String> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_max_open_files(10000);
        opts.set_use_fsync(false);
        opts.set_bytes_per_sync(8388608);
        opts.optimize_for_point_lookup(1024);
        opts.set_table_cache_num_shard_bits(6);
        opts.set_max_write_buffer_number(32);
        opts.set_write_buffer_size(536870912);
        opts.set_target_file_size_base(1073741824);
        opts.set_min_write_buffer_number_to_merge(4);
        opts.set_level_zero_stop_writes_trigger(2000);
        opts.set_level_zero_slowdown_writes_trigger(0);
        opts.set_compaction_style(rocksdb::DBCompactionStyle::Universal);
        opts.set_disable_auto_compactions(true);

        let db = DB::open(&opts, db_path)
            .map_err(|e| format!("Failed to open database: {}", e))?;

        let chain = Self::load_chain_from_db(&db)?;

        Ok(LedgerStorage {
            db: Arc::new(db),
            chain_lock: Arc::new(RwLock::new(chain)),
            pending_transactions: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Load existing blockchain from persistent storage
    fn load_chain_from_db(db: &DB) -> Result<Vec<Block>, String> {
        let mut chain = Vec::new();
        let iter = db.iterator(IteratorMode::Start);

        for item in iter {
            let (key, value) = item.map_err(|e| format!("Iterator error: {}", e))?;
            if key.starts_with(b"block_") {
                let block: Block = bincode::deserialize(&value)
                    .map_err(|e| format!("Deserialization error: {}", e))?;
                chain.push(block);
            }
        }

        chain.sort_by_key(|b| b.index);

        if chain.is_empty() {
            // Create genesis block
            let genesis = Block {
                index: 0,
                timestamp: Utc::now().timestamp(),
                data: b"Genesis Block - VitalStream Medical Blockchain".to_vec(),
                previous_hash: "0".to_string(),
                hash: String::new(),
                nonce: 0,
                validator: "system".to_string(),
            };
            chain.push(genesis);
        }

        Ok(chain)
    }

    /// Add new block to ledger with persistence
    pub fn add_block(&self, mut block: Block) -> Result<(), String> {
        // Calculate block hash
        block.hash = self.calculate_block_hash(&block);

        // Validate block
        self.validate_block(&block)?;

        // Write to persistent storage
        let key = format!("block_{:016x}", block.index);
        let value = bincode::serialize(&block)
            .map_err(|e| format!("Serialization error: {}", e))?;

        self.db.put(key.as_bytes(), &value)
            .map_err(|e| format!("Database write error: {}", e))?;

        // Update in-memory chain
        let mut chain = self.chain_lock.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        chain.push(block);

        Ok(())
    }

    /// Add transaction to pending pool
    pub fn add_transaction(&self, transaction: Transaction) -> Result<(), String> {
        // Validate transaction signature
        self.validate_transaction(&transaction)?;

        // Store transaction
        let key = format!("tx_{}", transaction.id);
        let value = bincode::serialize(&transaction)
            .map_err(|e| format!("Serialization error: {}", e))?;

        self.db.put(key.as_bytes(), &value)
            .map_err(|e| format!("Database write error: {}", e))?;

        // Add to pending pool
        let mut pending = self.pending_transactions.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        pending.push(transaction);

        Ok(())
    }

    /// Get block by index
    pub fn get_block(&self, index: u64) -> Result<Option<Block>, String> {
        let key = format!("block_{:016x}", index);
        match self.db.get(key.as_bytes()) {
            Ok(Some(value)) => {
                let block: Block = bincode::deserialize(&value)
                    .map_err(|e| format!("Deserialization error: {}", e))?;
                Ok(Some(block))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(format!("Database read error: {}", e)),
        }
    }

    /// Get transaction by ID
    pub fn get_transaction(&self, tx_id: &str) -> Result<Option<Transaction>, String> {
        let key = format!("tx_{}", tx_id);
        match self.db.get(key.as_bytes()) {
            Ok(Some(value)) => {
                let tx: Transaction = bincode::deserialize(&value)
                    .map_err(|e| format!("Deserialization error: {}", e))?;
                Ok(Some(tx))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(format!("Database read error: {}", e)),
        }
    }

    /// Get pending transactions
    pub fn get_pending_transactions(&self) -> Result<Vec<Transaction>, String> {
        let pending = self.pending_transactions.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        Ok(pending.clone())
    }

    /// Clear pending transactions (after block creation)
    pub fn clear_pending_transactions(&self) -> Result<(), String> {
        let mut pending = self.pending_transactions.write()
            .map_err(|e| format!("Lock error: {}", e))?;
        pending.clear();
        Ok(())
    }

    /// Get chain length
    pub fn get_chain_length(&self) -> Result<u64, String> {
        let chain = self.chain_lock.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        Ok(chain.len() as u64)
    }

    /// Get latest block
    pub fn get_latest_block(&self) -> Result<Block, String> {
        let chain = self.chain_lock.read()
            .map_err(|e| format!("Lock error: {}", e))?;
        chain.last()
            .cloned()
            .ok_or_else(|| "Chain is empty".to_string())
    }

    /// Validate entire chain integrity
    pub fn validate_chain(&self) -> Result<bool, String> {
        let chain = self.chain_lock.read()
            .map_err(|e| format!("Lock error: {}", e))?;

        for i in 1..chain.len() {
            let current = &chain[i];
            let previous = &chain[i - 1];

            // Verify hash
            if current.hash != self.calculate_block_hash(current) {
                return Ok(false);
            }

            // Verify chain linkage
            if current.previous_hash != previous.hash {
                return Ok(false);
            }

            // Verify index sequence
            if current.index != previous.index + 1 {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Calculate block hash
    fn calculate_block_hash(&self, block: &Block) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(block.index.to_le_bytes());
        hasher.update(block.timestamp.to_le_bytes());
        hasher.update(&block.data);
        hasher.update(block.previous_hash.as_bytes());
        hasher.update(block.nonce.to_le_bytes());
        hasher.update(block.validator.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Validate block
    fn validate_block(&self, block: &Block) -> Result<(), String> {
        let chain = self.chain_lock.read()
            .map_err(|e| format!("Lock error: {}", e))?;

        if let Some(last_block) = chain.last() {
            if block.index != last_block.index + 1 {
                return Err("Invalid block index".to_string());
            }
            if block.previous_hash != last_block.hash {
                return Err("Invalid previous hash".to_string());
            }
        }

        if block.hash != self.calculate_block_hash(block) {
            return Err("Invalid block hash".to_string());
        }

        Ok(())
    }

    /// Validate transaction with full signature verification
    fn validate_transaction(&self, transaction: &Transaction) -> Result<(), String> {
        // Basic validation
        if transaction.id.is_empty() {
            return Err("Transaction ID is empty".to_string());
        }
        if transaction.from.is_empty() {
            return Err("Transaction sender is empty".to_string());
        }
        if transaction.signature.is_empty() {
            return Err("Transaction signature is missing".to_string());
        }

        // Verify signature using Ed25519
        use ring::signature::{UnparsedPublicKey, ED25519};
        
        // Reconstruct message that was signed
        let mut message = Vec::new();
        message.extend_from_slice(transaction.id.as_bytes());
        message.extend_from_slice(transaction.from.as_bytes());
        message.extend_from_slice(transaction.to.as_bytes());
        message.extend_from_slice(transaction.data_type.as_bytes());
        message.extend_from_slice(transaction.data_hash.as_bytes());
        message.extend_from_slice(&transaction.timestamp.to_le_bytes());

        // Extract public key from 'from' address (base64 encoded)
        let public_key_bytes = base64::engine::general_purpose::STANDARD
            .decode(&transaction.from)
            .map_err(|e| format!("Invalid public key encoding: {}", e))?;

        if public_key_bytes.len() != 32 {
            return Err("Invalid public key length".to_string());
        }

        // Verify signature
        let public_key = UnparsedPublicKey::new(&ED25519, &public_key_bytes);
        public_key.verify(&message, &transaction.signature)
            .map_err(|_| "Invalid transaction signature".to_string())?;

        Ok(())
    }

    /// Compact database
    pub fn compact(&self) -> Result<(), String> {
        self.db.compact_range(None::<&[u8]>, None::<&[u8]>);
        Ok(())
    }

    /// Get database statistics
    pub fn get_stats(&self) -> Result<String, String> {
        self.db.property_value("rocksdb.stats")
            .map(|s| s.unwrap_or_else(|| "No stats available".to_string()))
            .map_err(|e| format!("Failed to get stats: {}", e))
    }

    /// Backup ledger to path
    pub fn backup(&self, backup_path: &str) -> Result<(), String> {
        let backup_engine = rocksdb::backup::BackupEngine::open(
            &Options::default(),
            Path::new(backup_path)
        ).map_err(|e| format!("Failed to open backup engine: {}", e))?;

        backup_engine.create_new_backup(&self.db)
            .map_err(|e| format!("Failed to create backup: {}", e))?;

        Ok(())
    }

    /// Restore ledger from backup
    pub fn restore(backup_path: &str, restore_path: &str) -> Result<(), String> {
        let mut backup_engine = rocksdb::backup::BackupEngine::open(
            &Options::default(),
            Path::new(backup_path)
        ).map_err(|e| format!("Failed to open backup engine: {}", e))?;

        let mut restore_opts = rocksdb::backup::RestoreOptions::default();
        restore_opts.set_keep_log_files(true);

        backup_engine.restore_from_latest_backup(
            Path::new(restore_path),
            Path::new(restore_path),
            &restore_opts
        ).map_err(|e| format!("Failed to restore backup: {}", e))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_ledger_creation() {
        let dir = tempdir().unwrap();
        let ledger = LedgerStorage::new(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(ledger.get_chain_length().unwrap(), 1); // Genesis block
    }

    #[test]
    fn test_add_block() {
        let dir = tempdir().unwrap();
        let ledger = LedgerStorage::new(dir.path().to_str().unwrap()).unwrap();
        
        let latest = ledger.get_latest_block().unwrap();
        let new_block = Block {
            index: latest.index + 1,
            timestamp: Utc::now().timestamp(),
            data: b"Test block".to_vec(),
            previous_hash: latest.hash.clone(),
            hash: String::new(),
            nonce: 0,
            validator: "test".to_string(),
        };

        ledger.add_block(new_block).unwrap();
        assert_eq!(ledger.get_chain_length().unwrap(), 2);
    }

    #[test]
    fn test_chain_validation() {
        let dir = tempdir().unwrap();
        let ledger = LedgerStorage::new(dir.path().to_str().unwrap()).unwrap();
        assert!(ledger.validate_chain().unwrap());
    }
}
