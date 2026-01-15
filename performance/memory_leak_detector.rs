//! Memory Leak Detection
//! Tools for detecting and preventing memory leaks

use std::sync::{Arc, Weak, Mutex};
use std::collections::HashMap;
use std::time::Instant;
use serde::{Deserialize, Serialize};

/// Memory leak detector using weak references
pub struct MemoryLeakDetector {
    tracked_objects: Arc<Mutex<HashMap<String, Vec<WeakRef>>>>,
}

impl MemoryLeakDetector {
    pub fn new() -> Self {
        Self {
            tracked_objects: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Track an object for potential leaks
    pub fn track<T>(&self, name: &str, obj: &Arc<T>) {
        let weak = Arc::downgrade(obj);
        let mut tracked = self.tracked_objects.lock().unwrap();
        
        tracked.entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(WeakRef {
                weak_ptr: Box::new(weak),
                created_at: Instant::now(),
            });
    }

    /// Check for leaked objects
    pub fn check_leaks(&self) -> LeakReport {
        let mut tracked = self.tracked_objects.lock().unwrap();
        let mut leaks = Vec::new();
        let mut total_tracked = 0;
        let mut total_alive = 0;

        for (name, refs) in tracked.iter_mut() {
            // Remove dead references
            refs.retain(|r| r.is_alive());
            
            let alive_count = refs.len();
            total_tracked += 1;
            total_alive += alive_count;

            if alive_count > 0 {
                let oldest = refs.iter()
                    .map(|r| r.created_at.elapsed().as_secs())
                    .max()
                    .unwrap_or(0);

                leaks.push(LeakInfo {
                    object_type: name.clone(),
                    alive_count,
                    oldest_age_secs: oldest,
                });
            }
        }

        LeakReport {
            total_tracked,
            total_alive,
            potential_leaks: leaks,
        }
    }

    /// Clear all tracked objects
    pub fn clear(&self) {
        let mut tracked = self.tracked_objects.lock().unwrap();
        tracked.clear();
    }
}

struct WeakRef {
    weak_ptr: Box<dyn WeakRefTrait>,
    created_at: Instant,
}

impl WeakRef {
    fn is_alive(&self) -> bool {
        self.weak_ptr.is_alive()
    }
}

trait WeakRefTrait {
    fn is_alive(&self) -> bool;
}

impl<T> WeakRefTrait for Weak<T> {
    fn is_alive(&self) -> bool {
        self.strong_count() > 0
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LeakReport {
    pub total_tracked: usize,
    pub total_alive: usize,
    pub potential_leaks: Vec<LeakInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LeakInfo {
    pub object_type: String,
    pub alive_count: usize,
    pub oldest_age_secs: u64,
}

impl LeakReport {
    pub fn print(&self) {
        println!("\n=== Memory Leak Report ===");
        println!("Total tracked types: {}", self.total_tracked);
        println!("Total alive objects: {}", self.total_alive);
        
        if self.potential_leaks.is_empty() {
            println!("No potential leaks detected.");
        } else {
            println!("\nPotential leaks:");
            for leak in &self.potential_leaks {
                println!("  - {}: {} alive objects (oldest: {}s)",
                    leak.object_type,
                    leak.alive_count,
                    leak.oldest_age_secs
                );
            }
        }
    }
}

/// Resource pool with automatic cleanup
pub struct ResourcePool<T> {
    resources: Arc<Mutex<Vec<Arc<T>>>>,
    max_size: usize,
}

impl<T> ResourcePool<T> {
    pub fn new(max_size: usize) -> Self {
        Self {
            resources: Arc::new(Mutex::new(Vec::new())),
            max_size,
        }
    }

    pub fn acquire(&self, resource: T) -> Arc<T> {
        let arc = Arc::new(resource);
        let mut resources = self.resources.lock().unwrap();
        
        // Clean up dead references
        resources.retain(|r| Arc::strong_count(r) > 1);
        
        // Add new resource if under limit
        if resources.len() < self.max_size {
            resources.push(arc.clone());
        }
        
        arc
    }

    pub fn get_active_count(&self) -> usize {
        let mut resources = self.resources.lock().unwrap();
        resources.retain(|r| Arc::strong_count(r) > 1);
        resources.len()
    }
}

/// Circular reference detector
pub struct CircularRefDetector {
    // Simplified implementation - in production, use petgraph
}

impl CircularRefDetector {
    pub fn new() -> Self {
        Self {}
    }

    pub fn detect_cycles(&self) -> bool {
        // Placeholder - implement with graph traversal
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_leak_detector() {
        let detector = MemoryLeakDetector::new();
        
        // Create some objects
        let obj1 = Arc::new("test1");
        let obj2 = Arc::new("test2");
        
        detector.track("String", &obj1);
        detector.track("String", &obj2);
        
        // Check leaks
        let report = detector.check_leaks();
        assert_eq!(report.total_alive, 2);
        
        // Drop one object
        drop(obj1);
        
        thread::sleep(Duration::from_millis(10));
        
        let report = detector.check_leaks();
        assert_eq!(report.total_alive, 1);
        
        report.print();
    }

    #[test]
    fn test_resource_pool() {
        let pool = ResourcePool::new(10);
        
        let r1 = pool.acquire("resource1");
        let r2 = pool.acquire("resource2");
        
        assert_eq!(pool.get_active_count(), 2);
        
        drop(r1);
        assert_eq!(pool.get_active_count(), 1);
        
        drop(r2);
        assert_eq!(pool.get_active_count(), 0);
    }
}
