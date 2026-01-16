//! Arrhythmia Detection Model
//!
//! ONNX-based arrhythmia classification with <10ms latency.

use crate::{ArrhythmiaType, InferenceError, Prediction, Result};
use ndarray::{Array1, Array2};
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder, Value};
use std::path::Path;
use std::time::Instant;
use tracing::{debug, info};

/// Arrhythmia detection model
pub struct ArrhythmiaDetector {
    session: Session,
    input_size: usize,
    num_classes: usize,
}

impl ArrhythmiaDetector {
    /// Create new arrhythmia detector from ONNX model
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to ONNX model file
    ///
    /// # Returns
    ///
    /// Initialized detector or error
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        info!("Loading arrhythmia detection model from {:?}", model_path.as_ref());

        // Create ONNX Runtime environment
        let environment = Environment::builder()
            .with_name("VitalStream")
            .build()
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        // Build session with optimizations
        let session = SessionBuilder::new(&environment)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?
            .with_model_from_file(model_path)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        info!("Model loaded successfully");

        Ok(Self {
            session,
            input_size: 360, // 1 second at 360 Hz
            num_classes: 15, // Extended arrhythmia classes
        })
    }

    /// Predict arrhythmia from ECG signal
    ///
    /// # Arguments
    ///
    /// * `signal` - Preprocessed ECG signal (normalized)
    ///
    /// # Returns
    ///
    /// Prediction with confidence scores
    pub fn predict(&self, signal: &[f32]) -> Result<Prediction> {
        // Validate input
        if signal.len() != self.input_size {
            return Err(InferenceError::InvalidInput(format!(
                "Expected signal length {}, got {}",
                self.input_size,
                signal.len()
            )));
        }

        let start = Instant::now();

        // Prepare input tensor (batch_size=1, channels=1, length=360)
        let input_array = Array2::from_shape_vec((1, self.input_size), signal.to_vec())
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;

        // Create ONNX tensor
        let input_tensor = Value::from_array(self.session.allocator(), &input_array)
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;

        // Run inference
        let outputs = self
            .session
            .run(vec![input_tensor])
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;

        // Extract output tensor
        let output_tensor = outputs[0]
            .try_extract::<f32>()
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;

        let probabilities: Vec<f32> = output_tensor.view().iter().copied().collect();

        // Find predicted class
        let (predicted_class, confidence) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, &prob)| (idx, prob))
            .unwrap();

        let inference_time = start.elapsed().as_secs_f32() * 1000.0;

        debug!(
            "Inference completed in {:.2}ms, predicted class: {}, confidence: {:.2}%",
            inference_time,
            predicted_class,
            confidence * 100.0
        );

        Ok(Prediction {
            arrhythmia_type: Self::index_to_arrhythmia(predicted_class),
            confidence,
            probabilities,
            inference_time_ms: inference_time,
        })
    }

    /// Batch prediction for multiple signals
    pub fn predict_batch(&self, signals: &[Vec<f32>]) -> Result<Vec<Prediction>> {
        signals.iter().map(|s| self.predict(s)).collect()
    }

    /// Convert class index to arrhythmia type
    fn index_to_arrhythmia(index: usize) -> ArrhythmiaType {
        match index {
            0 => ArrhythmiaType::Normal,
            1 => ArrhythmiaType::AtrialFibrillation,
            2 => ArrhythmiaType::AtrialFlutter,
            3 => ArrhythmiaType::SuperventricularTachycardia,
            4 => ArrhythmiaType::VentricularTachycardia,
            5 => ArrhythmiaType::VentricularFibrillation,
            6 => ArrhythmiaType::PrematureVentricularContraction,
            7 => ArrhythmiaType::PrematureAtrialContraction,
            8 => ArrhythmiaType::LeftBundleBranchBlock,
            9 => ArrhythmiaType::RightBundleBranchBlock,
            10 => ArrhythmiaType::Bradycardia,
            11 => ArrhythmiaType::Tachycardia,
            12 => ArrhythmiaType::MyocardialInfarction,
            13 => ArrhythmiaType::STEMI,
            14 => ArrhythmiaType::NSTEMI,
            _ => ArrhythmiaType::Normal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_conversion() {
        assert_eq!(
            ArrhythmiaDetector::index_to_arrhythmia(0),
            ArrhythmiaType::Normal
        );
        assert_eq!(
            ArrhythmiaDetector::index_to_arrhythmia(5),
            ArrhythmiaType::VentricularFibrillation
        );
    }
}
