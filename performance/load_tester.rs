//! Load Testing Framework
//! Tests system performance under 1000+ concurrent users

use tokio::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::{Semaphore, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestConfig {
    pub concurrent_users: usize,
    pub duration_secs: u64,
    pub ramp_up_secs: u64,
    pub target_url: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoadTestResult {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub avg_response_time_ms: f64,
    pub p50_response_time_ms: u64,
    pub p95_response_time_ms: u64,
    pub p99_response_time_ms: u64,
    pub requests_per_second: f64,
    pub errors: HashMap<String, u64>,
}

pub struct LoadTester {
    config: LoadTestConfig,
    client: reqwest::Client,
    results: Arc<RwLock<Vec<RequestResult>>>,
}

impl LoadTester {
    pub fn new(config: LoadTestConfig) -> Self {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(config.concurrent_users)
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            config,
            client,
            results: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Run load test
    pub async fn run(&self) -> LoadTestResult {
        println!("Starting load test with {} concurrent users...", self.config.concurrent_users);
        
        let start_time = Instant::now();
        let semaphore = Arc::new(Semaphore::new(self.config.concurrent_users));
        let mut handles = vec![];

        // Calculate requests per user
        let total_duration = Duration::from_secs(self.config.duration_secs);
        let requests_per_user = (self.config.duration_secs * 10) as usize; // 10 req/sec per user

        // Spawn concurrent users
        for user_id in 0..self.config.concurrent_users {
            let semaphore = semaphore.clone();
            let client = self.client.clone();
            let url = self.config.target_url.clone();
            let results = self.results.clone();
            let ramp_up = self.config.ramp_up_secs;

            let handle = tokio::spawn(async move {
                // Ramp-up delay
                let delay = (user_id as f64 / self.config.concurrent_users as f64) * ramp_up as f64;
                tokio::time::sleep(Duration::from_secs_f64(delay)).await;

                for _ in 0..requests_per_user {
                    let _permit = semaphore.acquire().await.unwrap();
                    
                    let req_start = Instant::now();
                    let result = client.get(&url).send().await;
                    let duration = req_start.elapsed();

                    let req_result = match result {
                        Ok(resp) => RequestResult {
                            success: resp.status().is_success(),
                            duration_ms: duration.as_millis() as u64,
                            error: if resp.status().is_success() {
                                None
                            } else {
                                Some(format!("HTTP {}", resp.status()))
                            },
                        },
                        Err(e) => RequestResult {
                            success: false,
                            duration_ms: duration.as_millis() as u64,
                            error: Some(e.to_string()),
                        },
                    };

                    results.write().await.push(req_result);
                    
                    // Small delay between requests
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            });

            handles.push(handle);
        }

        // Wait for all users to complete
        for handle in handles {
            let _ = handle.await;
        }

        let total_duration = start_time.elapsed();
        
        // Analyze results
        self.analyze_results(total_duration).await
    }

    async fn analyze_results(&self, total_duration: Duration) -> LoadTestResult {
        let results = self.results.read().await;
        
        let total_requests = results.len() as u64;
        let successful_requests = results.iter().filter(|r| r.success).count() as u64;
        let failed_requests = total_requests - successful_requests;

        // Calculate response times
        let mut durations: Vec<u64> = results.iter().map(|r| r.duration_ms).collect();
        durations.sort_unstable();

        let avg_response_time_ms = if !durations.is_empty() {
            durations.iter().sum::<u64>() as f64 / durations.len() as f64
        } else {
            0.0
        };

        let p50 = if !durations.is_empty() {
            durations[durations.len() / 2]
        } else {
            0
        };

        let p95 = if !durations.is_empty() {
            durations[(durations.len() * 95) / 100]
        } else {
            0
        };

        let p99 = if !durations.is_empty() {
            durations[(durations.len() * 99) / 100]
        } else {
            0
        };

        let requests_per_second = total_requests as f64 / total_duration.as_secs_f64();

        // Collect errors
        let mut errors = HashMap::new();
        for result in results.iter() {
            if let Some(error) = &result.error {
                *errors.entry(error.clone()).or_insert(0) += 1;
            }
        }

        LoadTestResult {
            total_requests,
            successful_requests,
            failed_requests,
            avg_response_time_ms,
            p50_response_time_ms: p50,
            p95_response_time_ms: p95,
            p99_response_time_ms: p99,
            requests_per_second,
            errors,
        }
    }
}

#[derive(Debug, Clone)]
struct RequestResult {
    success: bool,
    duration_ms: u64,
    error: Option<String>,
}

impl LoadTestResult {
    pub fn print(&self) {
        println!("\n=== Load Test Results ===");
        println!("Total Requests: {}", self.total_requests);
        println!("Successful: {} ({:.2}%)", 
            self.successful_requests,
            (self.successful_requests as f64 / self.total_requests as f64) * 100.0
        );
        println!("Failed: {} ({:.2}%)",
            self.failed_requests,
            (self.failed_requests as f64 / self.total_requests as f64) * 100.0
        );
        println!("\nResponse Times:");
        println!("  Average: {:.2}ms", self.avg_response_time_ms);
        println!("  P50: {}ms", self.p50_response_time_ms);
        println!("  P95: {}ms", self.p95_response_time_ms);
        println!("  P99: {}ms", self.p99_response_time_ms);
        println!("\nThroughput: {:.2} req/s", self.requests_per_second);
        
        if !self.errors.is_empty() {
            println!("\nErrors:");
            for (error, count) in &self.errors {
                println!("  - {}: {} occurrences", error, count);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires running server
    async fn test_load_tester() {
        let config = LoadTestConfig {
            concurrent_users: 100,
            duration_secs: 10,
            ramp_up_secs: 2,
            target_url: "http://localhost:8080/api/health".to_string(),
        };

        let tester = LoadTester::new(config);
        let result = tester.run().await;
        result.print();
        
        assert!(result.total_requests > 0);
    }
}
