//! Caching Strategy Implementation
//! Multi-layer caching for optimal performance

use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use lru::LruCache;
use std::num::NonZeroUsize;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedVitalSigns {
    pub patient_id: i32,
    pub heart_rate: i32,
    pub spo2: i32,
    pub timestamp: i64,
}

/// Multi-layer cache: L1 (in-memory LRU) -> L2 (Redis) -> L3 (Database)
pub struct CacheStrategy {
    // L1: In-memory LRU cache (fastest, smallest)
    l1_cache: Arc<RwLock<LruCache<String, CachedVitalSigns>>>,
    // L2: Redis cache (fast, larger)
    redis_client: redis::Client,
    // Cache TTLs
    l1_ttl: Duration,
    l2_ttl: Duration,
}

impl CacheStrategy {
    pub fn new(redis_url: &str) -> Result<Self, redis::RedisError> {
        let redis_client = redis::Client::open(redis_url)?;
        
        Ok(Self {
            l1_cache: Arc::new(RwLock::new(
                LruCache::new(NonZeroUsize::new(1000).unwrap())
            )),
            redis_client,
            l1_ttl: Duration::from_secs(30),
            l2_ttl: Duration::from_secs(300),
        })
    }

    /// Get from cache (L1 -> L2 -> miss)
    pub async fn get(&self, key: &str) -> Option<CachedVitalSigns> {
        // Try L1 cache first
        {
            let mut l1 = self.l1_cache.write().await;
            if let Some(value) = l1.get(key) {
                return Some(value.clone());
            }
        }

        // Try L2 (Redis) cache
        if let Ok(mut conn) = self.redis_client.get_async_connection().await {
            if let Ok(data) = conn.get::<_, String>(key).await {
                if let Ok(value) = serde_json::from_str::<CachedVitalSigns>(&data) {
                    // Promote to L1
                    let mut l1 = self.l1_cache.write().await;
                    l1.put(key.to_string(), value.clone());
                    return Some(value);
                }
            }
        }

        None
    }

    /// Set in all cache layers
    pub async fn set(&self, key: &str, value: CachedVitalSigns) -> Result<(), Box<dyn std::error::Error>> {
        // Set in L1
        {
            let mut l1 = self.l1_cache.write().await;
            l1.put(key.to_string(), value.clone());
        }

        // Set in L2 (Redis) with TTL
        let mut conn = self.redis_client.get_async_connection().await?;
        let serialized = serde_json::to_string(&value)?;
        conn.set_ex(key, serialized, self.l2_ttl.as_secs() as usize).await?;

        Ok(())
    }

    /// Invalidate cache entry
    pub async fn invalidate(&self, key: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Remove from L1
        {
            let mut l1 = self.l1_cache.write().await;
            l1.pop(key);
        }

        // Remove from L2
        let mut conn = self.redis_client.get_async_connection().await?;
        conn.del(key).await?;

        Ok(())
    }

    /// Invalidate pattern (e.g., "patient:123:*")
    pub async fn invalidate_pattern(&self, pattern: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut conn = self.redis_client.get_async_connection().await?;
        
        // Get all keys matching pattern
        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(pattern)
            .query_async(&mut conn)
            .await?;

        // Delete all matching keys
        if !keys.is_empty() {
            conn.del(&keys).await?;
        }

        // Clear L1 cache (simple approach - clear all)
        {
            let mut l1 = self.l1_cache.write().await;
            l1.clear();
        }

        Ok(())
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let l1_size = {
            let l1 = self.l1_cache.read().await;
            l1.len()
        };

        CacheStats {
            l1_entries: l1_size,
            l1_capacity: 1000,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CacheStats {
    pub l1_entries: usize,
    pub l1_capacity: usize,
}

/// Cache-aside pattern implementation
pub struct CacheAsidePattern {
    cache: Arc<CacheStrategy>,
}

impl CacheAsidePattern {
    pub fn new(cache: Arc<CacheStrategy>) -> Self {
        Self { cache }
    }

    /// Get with cache-aside pattern
    pub async fn get_or_fetch<F, Fut>(
        &self,
        key: &str,
        fetch_fn: F,
    ) -> Result<CachedVitalSigns, Box<dyn std::error::Error>>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<CachedVitalSigns, Box<dyn std::error::Error>>>,
    {
        // Try cache first
        if let Some(cached) = self.cache.get(key).await {
            return Ok(cached);
        }

        // Cache miss - fetch from source
        let value = fetch_fn().await?;
        
        // Store in cache
        self.cache.set(key, value.clone()).await?;
        
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires Redis connection
    async fn test_cache_strategy() {
        let cache = CacheStrategy::new("redis://127.0.0.1/").unwrap();
        
        let vital = CachedVitalSigns {
            patient_id: 123,
            heart_rate: 75,
            spo2: 98,
            timestamp: 1234567890,
        };

        cache.set("patient:123:latest", vital.clone()).await.unwrap();
        
        let retrieved = cache.get("patient:123:latest").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().patient_id, 123);
    }
}
