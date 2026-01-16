//! VitalStream ML Inference Engine
//!
//! Production-grade inference engine for ECG analysis and clinical decision support.
//! FDA/IEC 62304 compliant implementation in Rust.
//!
//! # Features
//!
//! - Low-latency inference (<10ms)
//! - Memory-safe execution
//! - Deterministic performance
//! - ONNX model support
//! - Clinical decision support algorithms

pub mod models;
pub mod preprocessing;
pub mod clinical;
pub mod utils;

use thiserror::Error;

/// Result type for inference operations
pub type Result<T> = std::result::Result<T, InferenceError>;

/// Inference error types
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Inference failed: {0}")]
    InferenceError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Preprocessing error: {0}")]
    PreprocessingError(String),

    #[error("Clinical algorithm error: {0}")]
    ClinicalError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Arrhythmia classification types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrhythmiaType {
    Normal,
    AtrialFibrillation,
    AtrialFlutter,
    SuperventricularTachycardia,
    VentricularTachycardia,
    VentricularFibrillation,
    PrematureVentricularContraction,
    PrematureAtrialContraction,
    LeftBundleBranchBlock,
    RightBundleBranchBlock,
    Bradycardia,
    Tachycardia,
    MyocardialInfarction,
    STEMI,
    NSTEMI,
}

impl ArrhythmiaType {
    /// Check if arrhythmia is critical (requires immediate attention)
    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            ArrhythmiaType::VentricularFibrillation
                | ArrhythmiaType::VentricularTachycardia
                | ArrhythmiaType::STEMI
        )
    }

    /// Get clinical priority (1=highest, 5=lowest)
    pub fn priority(&self) -> u8 {
        match self {
            ArrhythmiaType::VentricularFibrillation => 1,
            ArrhythmiaType::VentricularTachycardia => 1,
            ArrhythmiaType::STEMI => 1,
            ArrhythmiaType::MyocardialInfarction => 2,
            ArrhythmiaType::NSTEMI => 2,
            ArrhythmiaType::AtrialFibrillation => 3,
            ArrhythmiaType::SuperventricularTachycardia => 3,
            _ => 4,
        }
    }
}

/// Prediction result from ML model
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Predicted arrhythmia type
    pub arrhythmia_type: ArrhythmiaType,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// All class probabilities
    pub probabilities: Vec<f32>,
    /// Inference time in milliseconds
    pub inference_time_ms: f32,
}

impl Prediction {
    /// Check if prediction is reliable (confidence > threshold)
    pub fn is_reliable(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// Check if prediction requires clinical review
    pub fn requires_review(&self) -> bool {
        self.arrhythmia_type.is_critical() || self.confidence < 0.85
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arrhythmia_priority() {
        assert_eq!(ArrhythmiaType::VentricularFibrillation.priority(), 1);
        assert_eq!(ArrhythmiaType::Normal.priority(), 4);
    }

    #[test]
    fn test_critical_detection() {
        assert!(ArrhythmiaType::VentricularFibrillation.is_critical());
        assert!(!ArrhythmiaType::Normal.is_critical());
    }
}
