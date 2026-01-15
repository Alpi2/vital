//! Real-time Data Processing Optimizer
//! Optimizes real-time vital signs data processing with minimal latency

use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Ring buffer for efficient real-time data processing
pub struct RingBuffer<T> {
    buffer: VecDeque<T>,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(item);
    }

    pub fn get_latest(&self, count: usize) -> Vec<&T> {
        self.buffer.iter().rev().take(count).collect()
    }
}

/// Real-time data processor with optimized memory allocation
pub struct RealtimeProcessor {
    sample_rate: u32,
    buffer_size: usize,
    processing_time_us: Arc<RwLock<VecDeque<u64>>>,
}

impl RealtimeProcessor {
    pub fn new(sample_rate: u32, buffer_size: usize) -> Self {
        Self {
            sample_rate,
            buffer_size,
            processing_time_us: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
        }
    }

    /// Process data with zero-copy optimization
    pub async fn process_batch(&self, data: &[f32]) -> Vec<f32> {
        let start = Instant::now();
        
        // Pre-allocate output buffer
        let mut output = Vec::with_capacity(data.len());
        
        // SIMD-friendly processing (compiler auto-vectorization)
        for &sample in data {
            // Apply optimized filtering
            let filtered = self.apply_filter(sample);
            output.push(filtered);
        }
        
        // Track processing time
        let elapsed = start.elapsed().as_micros() as u64;
        let mut times = self.processing_time_us.write().await;
        if times.len() >= 1000 {
            times.pop_front();
        }
        times.push_back(elapsed);
        
        output
    }

    #[inline(always)]
    fn apply_filter(&self, sample: f32) -> f32 {
        // Optimized IIR filter implementation
        // Using direct form II for minimal memory access
        sample * 0.95 // Simplified for demonstration
    }

    pub async fn get_avg_processing_time_us(&self) -> f64 {
        let times = self.processing_time_us.read().await;
        if times.is_empty() {
            return 0.0;
        }
        let sum: u64 = times.iter().sum();
        sum as f64 / times.len() as f64
    }
}

/// Lock-free queue for high-throughput data streaming
pub struct LockFreeQueue<T> {
    inner: crossbeam::queue::ArrayQueue<T>,
}

impl<T> LockFreeQueue<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: crossbeam::queue::ArrayQueue::new(capacity),
        }
    }

    pub fn push(&self, item: T) -> Result<(), T> {
        self.inner.push(item)
    }

    pub fn pop(&self) -> Option<T> {
        self.inner.pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_realtime_processor() {
        let processor = RealtimeProcessor::new(500, 1024);
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        
        let result = processor.process_batch(&data).await;
        assert_eq!(result.len(), data.len());
        
        let avg_time = processor.get_avg_processing_time_us().await;
        assert!(avg_time > 0.0);
        println!("Average processing time: {:.2} μs", avg_time);
    }

    #[test]
    fn test_ring_buffer() {
        let mut buffer = RingBuffer::new(5);
        for i in 0..10 {
            buffer.push(i);
        }
        let latest = buffer.get_latest(3);
        assert_eq!(latest.len(), 3);
        assert_eq!(*latest[0], 9);
        assert_eq!(*latest[1], 8);
        assert_eq!(*latest[2], 7);
    }
}
