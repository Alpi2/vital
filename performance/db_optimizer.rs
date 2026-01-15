//! Database Query Optimizer
//! Optimizes PostgreSQL queries for vital signs data

use sqlx::{PgPool, Row};
use std::time::Instant;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryPerformance {
    pub query_name: String,
    pub execution_time_ms: f64,
    pub rows_returned: i64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct DatabaseOptimizer {
    pool: PgPool,
}

impl DatabaseOptimizer {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create optimized indexes for vital signs queries
    pub async fn create_indexes(&self) -> Result<(), sqlx::Error> {
        // Composite index for time-series queries
        sqlx::query(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vitals_patient_time 
             ON vital_signs(patient_id, timestamp DESC)"
        )
        .execute(&self.pool)
        .await?;

        // Partial index for active alarms
        sqlx::query(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alarms_active 
             ON alarms(patient_id, created_at DESC) 
             WHERE acknowledged = false"
        )
        .execute(&self.pool)
        .await?;

        // BRIN index for time-series data (space-efficient)
        sqlx::query(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vitals_timestamp_brin 
             ON vital_signs USING BRIN(timestamp)"
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Optimized query for recent vital signs (uses index)
    pub async fn get_recent_vitals_optimized(
        &self,
        patient_id: i32,
        limit: i64,
    ) -> Result<Vec<VitalSignRow>, sqlx::Error> {
        let start = Instant::now();
        
        let rows = sqlx::query_as::<_, VitalSignRow>(
            "SELECT id, patient_id, heart_rate, spo2, timestamp 
             FROM vital_signs 
             WHERE patient_id = $1 
             ORDER BY timestamp DESC 
             LIMIT $2"
        )
        .bind(patient_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        println!("Query executed in {:.2}ms, returned {} rows", elapsed, rows.len());
        
        Ok(rows)
    }

    /// Batch insert with COPY for high throughput
    pub async fn batch_insert_vitals(
        &self,
        vitals: Vec<VitalSignInsert>,
    ) -> Result<u64, sqlx::Error> {
        let start = Instant::now();
        
        // Use UNNEST for efficient batch insert
        let patient_ids: Vec<i32> = vitals.iter().map(|v| v.patient_id).collect();
        let heart_rates: Vec<i32> = vitals.iter().map(|v| v.heart_rate).collect();
        let spo2s: Vec<i32> = vitals.iter().map(|v| v.spo2).collect();
        
        let result = sqlx::query(
            "INSERT INTO vital_signs (patient_id, heart_rate, spo2, timestamp)
             SELECT * FROM UNNEST($1::int[], $2::int[], $3::int[], $4::timestamptz[])"
        )
        .bind(&patient_ids)
        .bind(&heart_rates)
        .bind(&spo2s)
        .bind(vec![chrono::Utc::now(); vitals.len()])
        .execute(&self.pool)
        .await?;

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        println!("Batch insert of {} rows in {:.2}ms", vitals.len(), elapsed);
        
        Ok(result.rows_affected())
    }

    /// Analyze query performance
    pub async fn analyze_query(&self, query: &str) -> Result<String, sqlx::Error> {
        let explain_query = format!("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {}", query);
        let row = sqlx::query(&explain_query)
            .fetch_one(&self.pool)
            .await?;
        
        let plan: String = row.try_get(0)?;
        Ok(plan)
    }

    /// Vacuum and analyze tables for optimal performance
    pub async fn maintain_tables(&self) -> Result<(), sqlx::Error> {
        sqlx::query("VACUUM ANALYZE vital_signs")
            .execute(&self.pool)
            .await?;
        
        sqlx::query("VACUUM ANALYZE alarms")
            .execute(&self.pool)
            .await?;
        
        Ok(())
    }
}

#[derive(Debug, sqlx::FromRow)]
pub struct VitalSignRow {
    pub id: i32,
    pub patient_id: i32,
    pub heart_rate: i32,
    pub spo2: i32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug)]
pub struct VitalSignInsert {
    pub patient_id: i32,
    pub heart_rate: i32,
    pub spo2: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires database connection
    async fn test_database_optimizer() {
        // This test requires a running PostgreSQL instance
        // Run with: cargo test --features integration-tests
    }
}
