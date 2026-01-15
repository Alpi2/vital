//! Code Profiling and Optimization Tools
//! Performance profiling utilities for identifying bottlenecks

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};

/// Function profiler with automatic timing
#[derive(Clone)]
pub struct Profiler {
    metrics: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Profile a function execution
    pub fn profile<F, R>(&self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        
        let mut metrics = self.metrics.lock().unwrap();
        metrics.entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
        
        result
    }

    /// Get profiling report
    pub fn get_report(&self) -> ProfilingReport {
        let metrics = self.metrics.lock().unwrap();
        let mut functions = Vec::new();

        for (name, durations) in metrics.iter() {
            if durations.is_empty() {
                continue;
            }

            let total: Duration = durations.iter().sum();
            let avg = total / durations.len() as u32;
            let min = *durations.iter().min().unwrap();
            let max = *durations.iter().max().unwrap();

            functions.push(FunctionMetrics {
                name: name.clone(),
                call_count: durations.len(),
                total_time_us: total.as_micros() as u64,
                avg_time_us: avg.as_micros() as u64,
                min_time_us: min.as_micros() as u64,
                max_time_us: max.as_micros() as u64,
            });
        }

        // Sort by total time (descending)
        functions.sort_by(|a, b| b.total_time_us.cmp(&a.total_time_us));

        ProfilingReport { functions }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.clear();
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProfilingReport {
    pub functions: Vec<FunctionMetrics>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionMetrics {
    pub name: String,
    pub call_count: usize,
    pub total_time_us: u64,
    pub avg_time_us: u64,
    pub min_time_us: u64,
    pub max_time_us: u64,
}

impl ProfilingReport {
    pub fn print(&self) {
        println!("\n=== Profiling Report ===");
        println!("{:<40} {:>10} {:>15} {:>15} {:>15} {:>15}",
            "Function", "Calls", "Total (μs)", "Avg (μs)", "Min (μs)", "Max (μs)");
        println!("{}", "-".repeat(120));
        
        for func in &self.functions {
            println!("{:<40} {:>10} {:>15} {:>15} {:>15} {:>15}",
                func.name,
                func.call_count,
                func.total_time_us,
                func.avg_time_us,
                func.min_time_us,
                func.max_time_us
            );
        }
    }
}

/// Memory profiler
pub struct MemoryProfiler {
    allocations: Arc<Mutex<Vec<AllocationRecord>>>,
}

impl MemoryProfiler {
    pub fn new() -> Self {
        Self {
            allocations: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn record_allocation(&self, size: usize, location: &str) {
        let mut allocations = self.allocations.lock().unwrap();
        allocations.push(AllocationRecord {
            size,
            location: location.to_string(),
            timestamp: Instant::now(),
        });
    }

    pub fn get_total_allocated(&self) -> usize {
        let allocations = self.allocations.lock().unwrap();
        allocations.iter().map(|a| a.size).sum()
    }

    pub fn get_allocation_report(&self) -> MemoryReport {
        let allocations = self.allocations.lock().unwrap();
        let mut by_location: HashMap<String, usize> = HashMap::new();

        for alloc in allocations.iter() {
            *by_location.entry(alloc.location.clone()).or_insert(0) += alloc.size;
        }

        let mut locations: Vec<_> = by_location.into_iter().collect();
        locations.sort_by(|a, b| b.1.cmp(&a.1));

        MemoryReport {
            total_allocated: self.get_total_allocated(),
            allocations_by_location: locations,
        }
    }
}

#[derive(Debug)]
struct AllocationRecord {
    size: usize,
    location: String,
    timestamp: Instant,
}

#[derive(Debug, Serialize)]
pub struct MemoryReport {
    pub total_allocated: usize,
    pub allocations_by_location: Vec<(String, usize)>,
}

/// Macro for easy profiling
#[macro_export]
macro_rules! profile {
    ($profiler:expr, $name:expr, $code:block) => {{
        $profiler.profile($name, || $code)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler() {
        let profiler = Profiler::new();

        // Profile some functions
        for _ in 0..10 {
            profiler.profile("fast_function", || {
                thread::sleep(Duration::from_micros(100));
            });
        }

        for _ in 0..5 {
            profiler.profile("slow_function", || {
                thread::sleep(Duration::from_micros(500));
            });
        }

        let report = profiler.get_report();
        assert_eq!(report.functions.len(), 2);
        
        report.print();
    }

    #[test]
    fn test_memory_profiler() {
        let profiler = MemoryProfiler::new();
        
        profiler.record_allocation(1024, "buffer_allocation");
        profiler.record_allocation(2048, "buffer_allocation");
        profiler.record_allocation(512, "small_allocation");

        let report = profiler.get_allocation_report();
        assert_eq!(report.total_allocated, 3584);
        assert_eq!(report.allocations_by_location.len(), 2);
    }
}
