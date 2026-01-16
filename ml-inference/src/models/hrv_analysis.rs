//! Heart Rate Variability (HRV) Analysis
//!
//! Time-domain and frequency-domain HRV metrics.

use crate::{InferenceError, Result};
use std::f32::consts::PI;

/// HRV analysis results
#[derive(Debug, Clone)]
pub struct HRVMetrics {
    // Time-domain metrics
    pub mean_rr: f32,      // Mean RR interval (ms)
    pub sdnn: f32,         // Standard deviation of NN intervals
    pub rmssd: f32,        // Root mean square of successive differences
    pub pnn50: f32,        // Percentage of successive RR intervals > 50ms
    
    // Frequency-domain metrics
    pub lf_power: f32,     // Low frequency power (0.04-0.15 Hz)
    pub hf_power: f32,     // High frequency power (0.15-0.4 Hz)
    pub lf_hf_ratio: f32,  // LF/HF ratio (sympathovagal balance)
    
    // Non-linear metrics
    pub sd1: f32,          // Poincaré plot SD1
    pub sd2: f32,          // Poincaré plot SD2
}

/// HRV analyzer
pub struct HRVAnalyzer {
    sampling_rate: f32,
}

impl HRVAnalyzer {
    pub fn new(sampling_rate: f32) -> Self {
        Self { sampling_rate }
    }

    /// Calculate HRV metrics from RR intervals
    ///
    /// # Arguments
    ///
    /// * `rr_intervals` - RR intervals in milliseconds
    pub fn analyze(&self, rr_intervals: &[f32]) -> Result<HRVMetrics> {
        if rr_intervals.len() < 10 {
            return Err(InferenceError::InvalidInput(
                "Need at least 10 RR intervals for HRV analysis".to_string(),
            ));
        }

        // Time-domain metrics
        let mean_rr = rr_intervals.iter().sum::<f32>() / rr_intervals.len() as f32;
        
        let variance: f32 = rr_intervals
            .iter()
            .map(|&rr| (rr - mean_rr).powi(2))
            .sum::<f32>()
            / rr_intervals.len() as f32;
        let sdnn = variance.sqrt();

        // RMSSD
        let successive_diffs: Vec<f32> = rr_intervals
            .windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .collect();
        let rmssd = (successive_diffs.iter().sum::<f32>() / successive_diffs.len() as f32).sqrt();

        // pNN50
        let nn50_count = rr_intervals
            .windows(2)
            .filter(|w| (w[1] - w[0]).abs() > 50.0)
            .count();
        let pnn50 = (nn50_count as f32 / (rr_intervals.len() - 1) as f32) * 100.0;

        // Frequency-domain (simplified)
        let (lf_power, hf_power) = self.calculate_frequency_domain(rr_intervals);
        let lf_hf_ratio = if hf_power > 0.0 {
            lf_power / hf_power
        } else {
            0.0
        };

        // Poincaré plot metrics
        let (sd1, sd2) = self.calculate_poincare(rr_intervals);

        Ok(HRVMetrics {
            mean_rr,
            sdnn,
            rmssd,
            pnn50,
            lf_power,
            hf_power,
            lf_hf_ratio,
            sd1,
            sd2,
        })
    }

    /// Calculate frequency-domain metrics (simplified FFT)
    fn calculate_frequency_domain(&self, rr_intervals: &[f32]) -> (f32, f32) {
        // TODO: Implement proper FFT-based frequency analysis
        // For now, return placeholder values
        (0.0, 0.0)
    }

    /// Calculate Poincaré plot metrics
    fn calculate_poincare(&self, rr_intervals: &[f32]) -> (f32, f32) {
        let diffs: Vec<f32> = rr_intervals
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        let sd1 = (diffs.iter().map(|&d| d.powi(2)).sum::<f32>() / diffs.len() as f32).sqrt()
            / 2.0_f32.sqrt();

        let variance: f32 = rr_intervals
            .iter()
            .map(|&rr| {
                let mean = rr_intervals.iter().sum::<f32>() / rr_intervals.len() as f32;
                (rr - mean).powi(2)
            })
            .sum::<f32>()
            / rr_intervals.len() as f32;

        let sd2 = (2.0 * variance - sd1.powi(2)).sqrt();

        (sd1, sd2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hrv_analysis() {
        let analyzer = HRVAnalyzer::new(360.0);
        let rr_intervals = vec![800.0, 820.0, 810.0, 830.0, 815.0, 825.0, 805.0, 835.0, 820.0, 810.0];
        
        let metrics = analyzer.analyze(&rr_intervals).unwrap();
        
        assert!(metrics.mean_rr > 0.0);
        assert!(metrics.sdnn > 0.0);
        assert!(metrics.rmssd > 0.0);
    }
}
