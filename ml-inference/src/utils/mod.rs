//! Utility functions

use chrono::{DateTime, Utc};

/// Normalize ECG signal to zero mean and unit variance
pub fn normalize_signal(signal: &[f32]) -> Vec<f32> {
    let mean = signal.iter().sum::<f32>() / signal.len() as f32;
    let std_dev = {
        let variance = signal
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / signal.len() as f32;
        variance.sqrt()
    };

    signal
        .iter()
        .map(|&x| (x - mean) / (std_dev + 1e-8))
        .collect()
}

/// Calculate heart rate from RR intervals
pub fn calculate_heart_rate(rr_intervals_ms: &[f32]) -> f32 {
    if rr_intervals_ms.is_empty() {
        return 0.0;
    }

    let mean_rr = rr_intervals_ms.iter().sum::<f32>() / rr_intervals_ms.len() as f32;
    60000.0 / mean_rr  // Convert to bpm
}

/// Timestamp for logging
pub fn current_timestamp() -> DateTime<Utc> {
    Utc::now()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_signal() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalize_signal(&signal);
        
        let mean = normalized.iter().sum::<f32>() / normalized.len() as f32;
        assert!(mean.abs() < 1e-6);
    }

    #[test]
    fn test_heart_rate_calculation() {
        let rr_intervals = vec![800.0, 820.0, 810.0];  // ms
        let hr = calculate_heart_rate(&rr_intervals);
        assert!((hr - 74.0).abs() < 1.0);  // ~74 bpm
    }
}
