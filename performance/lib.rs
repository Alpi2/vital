//! VitalStream Performance Optimization Library
//! 
//! This library provides comprehensive performance optimization tools for the VitalStream project.
//! It includes real-time data processing optimization, database query optimization, caching strategies,
//! network latency reduction, code profiling, memory leak detection, and load testing capabilities.

pub mod realtime_optimizer;
pub mod db_optimizer;
pub mod cache_strategy;
pub mod network_optimizer;
pub mod profiler;
pub mod memory_leak_detector;
pub mod load_tester;

pub use realtime_optimizer::{RealtimeProcessor, RingBuffer, LockFreeQueue};
pub use db_optimizer::DatabaseOptimizer;
pub use cache_strategy::{CacheStrategy, CacheAsidePattern};
pub use network_optimizer::{WebSocketOptimizer, Http2ConnectionPool, NetworkMonitor};
pub use profiler::{Profiler, MemoryProfiler};
pub use memory_leak_detector::{MemoryLeakDetector, ResourcePool};
pub use load_tester::{LoadTester, LoadTestConfig};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_exports() {
        // Verify all modules are accessible
        let _profiler = Profiler::new();
        let _mem_profiler = MemoryProfiler::new();
        let _leak_detector = MemoryLeakDetector::new();
    }
}
