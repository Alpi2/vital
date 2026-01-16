//! Signal quality assessment

use crate::{InferenceError, Result};

/// Signal quality metrics
#[derive(Debug, Clone)]
pub struct SignalQuality {
    pub sqi: f32,           // Signal Quality Index (0-1)
    pub snr: f32,           // Signal-to-Noise Ratio (dB)
    pub is_valid: bool,     // Quality above threshold
    pub artifacts: Vec<ArtifactType>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactType {
    Saturation,
    FlatLine,
    ExcessiveNoise,
    MotionArtifact,
    LeadOff,
}

pub struct SignalQualityAssessor {
    sqi_threshold: f32,
}

impl SignalQualityAssessor {
    pub fn new(sqi_threshold: f32) -> Self {
        Self { sqi_threshold }
    }

    /// Assess signal quality
    pub fn assess(&self, signal: &[f32]) -> Result<SignalQuality> {
        let sqi = self.calculate_sqi(signal);
        let snr = self.calculate_snr(signal);
        let artifacts = self.detect_artifacts(signal);
        let is_valid = sqi >= self.sqi_threshold && artifacts.is_empty();

        Ok(SignalQuality {
            sqi,
            snr,
            is_valid,
            artifacts,
        })
    }

    fn calculate_sqi(&self, signal: &[f32]) -> f32 {
        // Simplified SQI calculation
        let mean = signal.iter().sum::<f32>() / signal.len() as f32;
        let variance = signal
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / signal.len() as f32;
        
        // Normalize to 0-1
        (variance / (variance + 1.0)).min(1.0)
    }

    fn calculate_snr(&self, signal: &[f32]) -> f32 {
        // Simplified SNR calculation
        let signal_power = signal.iter().map(|&x| x.powi(2)).sum::<f32>() / signal.len() as f32;
        let noise_estimate = signal
            .windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum::<f32>()
            / (signal.len() - 1) as f32;
        
        10.0 * (signal_power / noise_estimate).log10()
    }

    fn detect_artifacts(&self, signal: &[f32]) -> Vec<ArtifactType> {
        let mut artifacts = Vec::new();

        // Flat line detection
        let std_dev = {
            let mean = signal.iter().sum::<f32>() / signal.len() as f32;
            (signal
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>()
                / signal.len() as f32)
                .sqrt()
        };

        if std_dev < 0.01 {
            artifacts.push(ArtifactType::FlatLine);
        }

        // Saturation detection
        let max_val = signal.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_val = signal.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        
        if (max_val - min_val).abs() > 10.0 {
            artifacts.push(ArtifactType::Saturation);
        }

        artifacts
    }
}
