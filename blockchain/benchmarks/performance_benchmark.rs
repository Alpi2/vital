// Blockchain Performance Benchmarks
// Comprehensive TPS, latency, and throughput testing

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use serde::{Serialize, Deserialize};
use sha3::{Sha3_256, Digest};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Transaction {
    id: String,
    from: String,
    to: String,
    data: Vec<u8>,
    timestamp: i64,
    signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Block {
    index: u64,
    timestamp: i64,
    transactions: Vec<Transaction>,
    previous_hash: String,
    hash: String,
    nonce: u64,
}

fn generate_transaction(id: usize) -> Transaction {
    Transaction {
        id: format!("tx_{}", id),
        from: format!("addr_{}", id % 100),
        to: format!("addr_{}", (id + 1) % 100),
        data: vec![0u8; 256],
        timestamp: chrono::Utc::now().timestamp(),
        signature: vec![0u8; 64],
    }
}

fn calculate_hash(block: &Block) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(block.index.to_le_bytes());
    hasher.update(block.timestamp.to_le_bytes());
    hasher.update(&bincode::serialize(&block.transactions).unwrap());
    hasher.update(block.previous_hash.as_bytes());
    hasher.update(block.nonce.to_le_bytes());
    format!("{:x}", hasher.finalize())
}

fn create_block(index: u64, tx_count: usize, previous_hash: String) -> Block {
    let transactions: Vec<Transaction> = (0..tx_count)
        .map(|i| generate_transaction(i))
        .collect();

    let mut block = Block {
        index,
        timestamp: chrono::Utc::now().timestamp(),
        transactions,
        previous_hash,
        hash: String::new(),
        nonce: 0,
    };

    block.hash = calculate_hash(&block);
    block
}

fn benchmark_transaction_creation(c: &mut Criterion) {
    c.bench_function("transaction_creation", |b| {
        b.iter(|| {
            black_box(generate_transaction(0));
        });
    });
}

fn benchmark_transaction_serialization(c: &mut Criterion) {
    let tx = generate_transaction(0);
    
    c.bench_function("transaction_serialization", |b| {
        b.iter(|| {
            black_box(bincode::serialize(&tx).unwrap());
        });
    });
}

fn benchmark_block_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_creation");
    
    for tx_count in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(tx_count),
            tx_count,
            |b, &tx_count| {
                b.iter(|| {
                    black_box(create_block(1, tx_count, "genesis".to_string()));
                });
            },
        );
    }
    group.finish();
}

fn benchmark_hash_calculation(c: &mut Criterion) {
    let block = create_block(1, 100, "genesis".to_string());
    
    c.bench_function("hash_calculation", |b| {
        b.iter(|| {
            black_box(calculate_hash(&block));
        });
    });
}

fn benchmark_block_validation(c: &mut Criterion) {
    let block = create_block(1, 100, "genesis".to_string());
    
    c.bench_function("block_validation", |b| {
        b.iter(|| {
            let calculated_hash = calculate_hash(&block);
            black_box(calculated_hash == block.hash);
        });
    });
}

fn benchmark_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.measurement_time(Duration::from_secs(10));
    
    for block_count in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(block_count),
            block_count,
            |b, &block_count| {
                b.iter(|| {
                    let mut previous_hash = "genesis".to_string();
                    for i in 0..block_count {
                        let block = create_block(i as u64, 100, previous_hash.clone());
                        previous_hash = block.hash.clone();
                        black_box(block);
                    }
                });
            },
        );
    }
    group.finish();
}

fn benchmark_concurrent_transactions(c: &mut Criterion) {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let mut group = c.benchmark_group("concurrent_transactions");
    
    for thread_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(thread_count),
            thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    let transactions = Arc::new(Mutex::new(Vec::new()));
                    let mut handles = vec![];

                    for _ in 0..thread_count {
                        let txs = Arc::clone(&transactions);
                        let handle = thread::spawn(move || {
                            for i in 0..100 {
                                let tx = generate_transaction(i);
                                txs.lock().unwrap().push(tx);
                            }
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    black_box(transactions);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_transaction_creation,
    benchmark_transaction_serialization,
    benchmark_block_creation,
    benchmark_hash_calculation,
    benchmark_block_validation,
    benchmark_throughput,
    benchmark_concurrent_transactions
);
criterion_main!(benches);
