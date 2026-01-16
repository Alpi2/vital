//! ML Model wrappers for ONNX inference

pub mod arrhythmia;
pub mod mi_detection;
pub mod hrv_analysis;

pub use arrhythmia::ArrhythmiaDetector;
pub use mi_detection::MIDetector;
pub use hrv_analysis::HRVAnalyzer;
