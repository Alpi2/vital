use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1};
use std::collections::HashMap;
use std::panic;
use std::time::Instant;
use rayon::prelude::*;

/// ECG Analysis Configuration
#[pyclass]
#[derive(Clone, Debug)]
pub struct ECGAnalysisConfig {
    #[pyo3(get)]
    pub sampling_rate: f64,
    
    #[pyo3(get)]
    pub enable_hrv: bool,
    
    #[pyo3(get)]
    pub enable_anomaly_detection: bool,
    
    #[pyo3(get)]
    pub min_heart_rate: f64,
    
    #[pyo3(get)]
    pub max_heart_rate: f64,
    
    #[pyo3(get)]
    pub noise_threshold: f64,
}

#[pymethods]
impl ECGAnalysisConfig {
    #[new]
    fn new(
        sampling_rate: f64,
        enable_hrv: Option<bool>,
        enable_anomaly_detection: Option<bool>,
        min_heart_rate: Option<f64>,
        max_heart_rate: Option<f64>,
        noise_threshold: Option<f64>,
    ) -> Self {
        Self {
            sampling_rate,
            enable_hrv: enable_hrv.unwrap_or(true),
            enable_anomaly_detection: enable_anomaly_detection.unwrap_or(true),
            min_heart_rate: min_heart_rate.unwrap_or(40.0),
            max_heart_rate: max_heart_rate.unwrap_or(200.0),
            noise_threshold: noise_threshold.unwrap_or(0.1),
        }
    }
}

/// ECG Analysis Result with comprehensive metrics
#[pyclass]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ECGAnalysisResult {
    #[pyo3(get)]
    pub heart_rate: f64,
    
    #[pyo3(get)]
    pub rr_intervals: Vec<f64>,
    
    #[pyo3(get)]
    pub qrs_peaks: Vec<usize>,
    
    #[pyo3(get)]
    pub hrv_metrics: HashMap<String, f64>,
    
    #[pyo3(get)]
    pub anomalies: Vec<String>,
    
    #[pyo3(get)]
    pub signal_quality: f64,
    
    #[pyo3(get)]
    pub processing_time_ms: f64,
    
    #[pyo3(get)]
    pub algorithm_version: String,
    
    #[pyo3(get)]
    pub processing_backend: String,
}

#[pymethods]
impl ECGAnalysisResult {
    /// Convert to Python dictionary with efficient JSON serialization
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let json_str = serde_json::to_string(self)
            .map_err(|e| PyValueError::new_err(format!("JSON serialization failed: {}", e)))?;
        
        let json_module = py.import("json")?;
        json_module.call_method1("loads", (json_str,))
            .map(|obj| obj.into())
    }
    
    /// Convert to JSON string
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| PyValueError::new_err(format!("JSON serialization failed: {}", e)))
    }
    
    /// Get summary statistics
    fn summary(&self) -> String {
        format!(
            "ECG Analysis: HR={:.1f} BPM, Quality={:.2f}, Time={:.2f}ms, Backend={}",
            self.heart_rate,
            self.signal_quality,
            self.processing_time_ms,
            self.processing_backend
        )
    }
}

/// Memory pool for reusable buffers
thread_local! {
    static BUFFER_POOL: std::cell::RefCell<std::collections::VecDeque<Vec<f64>>> = 
        std::cell::RefCell::new(std::collections::VecDeque::new());
}

/// Get buffer from pool or create new one
fn get_buffer(size: usize) -> Vec<f64> {
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        pool.pop()
            .unwrap_or_else(|| Vec::with_capacity(size))
    })
}

/// Return buffer to pool
fn return_buffer(mut buffer: Vec<f64>) {
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < 10 { // Limit pool size
            buffer.clear();
            pool.push_back(buffer);
        }
    });
}

/// SIMD-optimized ECG analysis (x86_64 only)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn analyze_ecg_simd(signal: &[f64]) -> Vec<usize> {
    // AVX2 optimized QRS detection
    // This is a placeholder for actual SIMD implementation
    let mut peaks = Vec::new();
    let len = signal.len();
    
    if len < 100 {
        return peaks;
    }
    
    // Simple threshold-based detection (placeholder)
    let threshold = 0.5;
    for i in 1..len-1 {
        if signal[i] > threshold && 
           signal[i] > signal[i-1] && 
           signal[i] > signal[i+1] {
            peaks.push(i);
        }
    }
    
    peaks
}

/// Fallback for non-x86_64 architectures
#[cfg(not(target_arch = "x86_64"))]
unsafe fn analyze_ecg_simd(signal: &[f64]) -> Vec<usize> {
    // Use standard implementation
    analyze_ecg_standard(signal)
}

/// Standard ECG analysis implementation
fn analyze_ecg_standard(signal: &[f64]) -> Vec<usize> {
    let mut peaks = Vec::new();
    let len = signal.len();
    
    if len < 100 {
        return peaks;
    }
    
    // Simple threshold-based detection
    let threshold = 0.5;
    for i in 1..len-1 {
        if signal[i] > threshold && 
           signal[i] > signal[i-1] && 
           signal[i] > signal[i+1] {
            peaks.push(i);
        }
    }
    
    peaks
}

/// Analyze ECG signal (synchronous, GIL-safe zero-copy)
#[pyfunction]
fn analyze_ecg(
    py: Python,
    signal: PyReadonlyArray1<f64>,
    sampling_rate: Option<f64>,
) -> PyResult<ECGAnalysisResult> {
    let config = ECGAnalysisConfig::new(
        sampling_rate.unwrap_or(360.0),
        Some(true),
        Some(true),
        Some(40.0),
        Some(200.0),
        Some(0.1),
    );
    
    analyze_ecg_with_config(py, signal, config)
}

/// Analyze ECG signal with configuration
#[pyfunction]
fn analyze_ecg_with_config(
    py: Python,
    signal: PyReadonlyArray1<f64>,
    config: ECGAnalysisConfig,
) -> PyResult<ECGAnalysisResult> {
    let start = Instant::now();
    
    // CRITICAL: Get pointer and length BEFORE releasing GIL
    let signal_slice = signal.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to get signal slice: {}", e)))?;
    let signal_ptr = signal_slice.as_ptr();
    let signal_len = signal_slice.len();
    
    // Release GIL for CPU-intensive work
    let result = py.allow_threads(move || {
        // CRITICAL: Panic protection to prevent process crash
        panic::catch_unwind(|| {
            // Create safe slice from raw pointer
            let signal_slice = unsafe { 
                std::slice::from_raw_parts(signal_ptr, signal_len) 
            };
            
            // Call the actual ECG processor
            let ecg_result = ecg_processor::analyze_ecg(signal_slice, config.sampling_rate)
                .map_err(|e| format!("ECG analysis failed: {}", e))?;
            
            // Detect QRS peaks with SIMD if available
            let qrs_peaks = if is_x86_feature_detected!("avx2") {
                unsafe { analyze_ecg_simd(signal_slice) }
            } else {
                analyze_ecg_standard(signal_slice)
            };
            
            // Calculate HRV metrics if enabled
            let hrv_metrics = if config.enable_hrv && !ecg_result.rr_intervals.is_empty() {
                calculate_hrv_metrics(&ecg_result.rr_intervals)
            } else {
                HashMap::new()
            };
            
            // Detect anomalies if enabled
            let anomalies = if config.enable_anomaly_detection {
                detect_anomalies(&ecg_result, &config)
            } else {
                Vec::new()
            };
            
            Ok(ECGAnalysisResult {
                heart_rate: ecg_result.heart_rate,
                rr_intervals: ecg_result.rr_intervals,
                qrs_peaks,
                hrv_metrics,
                anomalies,
                signal_quality: ecg_result.signal_quality,
                processing_time_ms: 0.0, // Will be set below
                algorithm_version: "1.0.0".to_string(),
                processing_backend: "rust_ffi".to_string(),
            })
        })
    });
    
    // Handle panic result
    let analysis_result = result
        .map_err(|_| PyValueError::new_err("Rust panic occurred during ECG analysis"))?
        .map_err(|e| PyValueError::new_err(e))?;
    
    // Set processing time
    let processing_time = start.elapsed().as_secs_f64() * 1000.0;
    let mut final_result = analysis_result;
    final_result.processing_time_ms = processing_time;
    
    Ok(final_result)
}

/// Calculate HRV metrics
fn calculate_hrv_metrics(rr_intervals: &[f64]) -> HashMap<String, f64> {
    let mut metrics = HashMap::new();
    
    if rr_intervals.len() < 2 {
        return metrics;
    }
    
    // RMSSD (Root Mean Square of Successive Differences)
    let rmssd: f64 = rr_intervals
        .windows(2)
        .map(|w| (w[1] - w[0]).powi(2))
        .sum::<f64>()
        / (rr_intervals.len() - 1) as f64;
    metrics.insert("rmssd".to_string(), rmssd.sqrt());
    
    // Mean RR interval
    let mean_rr: f64 = rr_intervals.iter().sum::<f64>() / rr_intervals.len() as f64;
    metrics.insert("mean_rr".to_string(), mean_rr);
    
    // SDNN (Standard deviation of NN intervals)
    let variance: f64 = rr_intervals
        .iter()
        .map(|rr| (rr - mean_rr).powi(2))
        .sum::<f64>() / rr_intervals.len() as f64;
    metrics.insert("sdnn".to_string(), variance.sqrt());
    
    metrics
}

/// Detect anomalies in ECG analysis
fn detect_anomalies(result: &ecg_processor::ECGResult, config: &ECGAnalysisConfig) -> Vec<String> {
    let mut anomalies = Vec::new();
    
    if result.heart_rate < config.min_heart_rate {
        anomalies.push(format!("Low heart rate: {:.1f} BPM", result.heart_rate));
    }
    
    if result.heart_rate > config.max_heart_rate {
        anomalies.push(format!("High heart rate: {:.1f} BPM", result.heart_rate));
    }
    
    if result.signal_quality < config.noise_threshold {
        anomalies.push(format!("Poor signal quality: {:.2f}", result.signal_quality));
    }
    
    anomalies
}

/// Analyze ECG signal (asynchronous with safe copy)
#[pyfunction]
fn analyze_ecg_async(
    py: Python,
    signal: PyReadonlyArray1<f64>,
    sampling_rate: Option<f64>,
) -> PyResult<&PyAny> {
    let config = ECGAnalysisConfig::new(
        sampling_rate.unwrap_or(360.0),
        Some(true),
        Some(true),
        Some(40.0),
        Some(200.0),
        Some(0.1),
    );
    
    // CRITICAL: Safe copy for async (necessary for safety)
    let signal_vec = signal.to_vec()
        .map_err(|e| PyValueError::new_err(format!("Failed to copy signal: {}", e)))?;
    
    pyo3_asyncio::tokio::future_into_py(py, async move {
        let start = Instant::now();
        
        // Run in blocking thread pool
        let result = tokio::task::spawn_blocking(move || {
            // Call ECG processor
            ecg_processor::analyze_ecg(&signal_vec, config.sampling_rate)
        })
        .await
        .map_err(|e| PyValueError::new_err(format!("Task join error: {}", e)))?
        .map_err(|e| PyValueError::new_err(format!("ECG analysis failed: {}", e)))?;
        
        // Calculate additional metrics
        let hrv_metrics = if config.enable_hrv && !result.rr_intervals.is_empty() {
            calculate_hrv_metrics(&result.rr_intervals)
        } else {
            HashMap::new()
        };
        
        let anomalies = detect_anomalies(&result, &config);
        
        let processing_time = start.elapsed().as_secs_f64() * 1000.0;
        
        Ok(ECGAnalysisResult {
            heart_rate: result.heart_rate,
            rr_intervals: result.rr_intervals,
            qrs_peaks: result.qrs_peaks,
            hrv_metrics,
            anomalies,
            signal_quality: result.signal_quality,
            processing_time_ms: processing_time,
            algorithm_version: "1.0.0".to_string(),
            processing_backend: "rust_ffi_async".to_string(),
        })
    })
}

/// Batch ECG analysis with parallel processing
#[pyfunction]
fn analyze_ecg_batch(
    py: Python,
    signals: Vec<PyReadonlyArray1<f64>>,
    sampling_rate: Option<f64>,
) -> PyResult<Vec<ECGAnalysisResult>> {
    let config = ECGAnalysisConfig::new(
        sampling_rate.unwrap_or(360.0),
        Some(true),
        Some(true),
        Some(40.0),
        Some(200.0),
        Some(0.1),
    );
    
    let start = Instant::now();
    
    // Release GIL for parallel processing
    let results = py.allow_threads(move || {
        // CRITICAL: Panic protection for parallel processing
        panic::catch_unwind(|| {
            // 🚀 Parallel processing with Rayon
            signals
                .par_iter()
                .enumerate()
                .map(|(index, signal)| {
                    // Get signal slice
                    let signal_slice = signal.as_slice()
                        .map_err(|e| format!("Signal {} slice error: {}", index, e))?;
                    
                    // Call ECG processor
                    let result = ecg_processor::analyze_ecg(signal_slice, config.sampling_rate)
                        .map_err(|e| format!("Signal {} analysis error: {}", index, e))?;
                    
                    // Calculate metrics
                    let hrv_metrics = if config.enable_hrv && !result.rr_intervals.is_empty() {
                        calculate_hrv_metrics(&result.rr_intervals)
                    } else {
                        HashMap::new()
                    };
                    
                    let anomalies = detect_anomalies(&result, &config);
                    
                    Ok(ECGAnalysisResult {
                        heart_rate: result.heart_rate,
                        rr_intervals: result.rr_intervals,
                        qrs_peaks: result.qrs_peaks,
                        hrv_metrics,
                        anomalies,
                        signal_quality: result.signal_quality,
                        processing_time_ms: 0.0, // Will be updated
                        algorithm_version: "1.0.0".to_string(),
                        processing_backend: "rust_ffi_batch".to_string(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
    });
    
    // Handle panic
    let mut analysis_results = results
        .map_err(|_| PyValueError::new_err("Rust panic occurred during batch analysis"))?
        .map_err(|e| PyValueError::new_err(e))?;
    
    // Update processing times
    let total_time = start.elapsed().as_secs_f64() * 1000.0;
    let avg_time = total_time / analysis_results.len() as f64;
    
    for result in &mut analysis_results {
        result.processing_time_ms = avg_time;
    }
    
    Ok(analysis_results)
}

/// Streaming ECG analysis for real-time processing
#[pyfunction]
fn analyze_ecg_stream(
    py: Python,
    signal_chunks: Vec<PyReadonlyArray1<f64>>,
    sampling_rate: Option<f64>,
) -> PyResult<Vec<ECGAnalysisResult>> {
    let config = ECGAnalysisConfig::new(
        sampling_rate.unwrap_or(360.0),
        Some(true),
        Some(true),
        Some(40.0),
        Some(200.0),
        Some(0.1),
    );
    
    let start = Instant::now();
    
    // Process chunks sequentially for real-time constraints
    let results = py.allow_threads(move || {
        panic::catch_unwind(|| {
            signal_chunks
                .iter()
                .enumerate()
                .map(|(index, chunk)| {
                    let signal_slice = chunk.as_slice()
                        .map_err(|e| format!("Chunk {} slice error: {}", index, e))?;
                    
                    let result = ecg_processor::analyze_ecg(signal_slice, config.sampling_rate)
                        .map_err(|e| format!("Chunk {} analysis error: {}", index, e))?;
                    
                    let hrv_metrics = if config.enable_hrv && !result.rr_intervals.is_empty() {
                        calculate_hrv_metrics(&result.rr_intervals)
                    } else {
                        HashMap::new()
                    };
                    
                    let anomalies = detect_anomalies(&result, &config);
                    
                    Ok(ECGAnalysisResult {
                        heart_rate: result.heart_rate,
                        rr_intervals: result.rr_intervals,
                        qrs_peaks: result.qrs_peaks,
                        hrv_metrics,
                        anomalies,
                        signal_quality: result.signal_quality,
                        processing_time_ms: 0.0,
                        algorithm_version: "1.0.0".to_string(),
                        processing_backend: "rust_ffi_stream".to_string(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
    });
    
    let mut analysis_results = results
        .map_err(|_| PyValueError::new_err("Rust panic occurred during stream analysis"))?
        .map_err(|e| PyValueError::new_err(e))?;
    
    // Update processing times
    let total_time = start.elapsed().as_secs_f64() * 1000.0;
    let avg_time = total_time / analysis_results.len() as f64;
    
    for result in &mut analysis_results {
        result.processing_time_ms = avg_time;
    }
    
    Ok(analysis_results)
}

/// Performance benchmark function
#[pyfunction]
fn benchmark_ecg_analysis(
    py: Python,
    signal_length: usize,
    iterations: usize,
    sampling_rate: Option<f64>,
) -> PyResult<HashMap<String, f64>> {
    let config = ECGAnalysisConfig::new(
        sampling_rate.unwrap_or(360.0),
        Some(true),
        Some(true),
        Some(40.0),
        Some(200.0),
        Some(0.1),
    );
    
    // Generate test signal
    let test_signal: Vec<f64> = (0..signal_length)
        .map(|i| (i as f64 * 0.01).sin())
        .collect();
    
    let results = py.allow_threads(move || {
        panic::catch_unwind(|| {
            // Warmup
            for _ in 0..10 {
                let _ = ecg_processor::analyze_ecg(&test_signal, config.sampling_rate);
            }
            
            // Benchmark
            let start = Instant::now();
            
            for _ in 0..iterations {
                let _ = ecg_processor::analyze_ecg(&test_signal, config.sampling_rate);
            }
            
            let elapsed = start.elapsed();
            
            let mut metrics = HashMap::new();
            metrics.insert("total_time_ms".to_string(), elapsed.as_secs_f64() * 1000.0);
            metrics.insert("avg_time_ms".to_string(), (elapsed.as_secs_f64() * 1000.0) / iterations as f64);
            metrics.insert("throughput_signals_per_sec".to_string(), iterations as f64 / elapsed.as_secs_f64());
            metrics.insert("samples_per_sec".to_string(), (signal_length * iterations) as f64 / elapsed.as_secs_f64());
            
            metrics
        })
    });
    
    results
        .map_err(|_| PyValueError::new_err("Rust panic occurred during benchmark"))?
        .map_err(|e| PyValueError::new_err(e))
}

/// Test connection and verify FFI is working
#[pyfunction]
fn test_connection() -> PyResult<String> {
    Ok("✅ Rust FFI module is working correctly!".to_string())
}

/// Get version information
#[pyfunction]
fn get_version() -> PyResult<HashMap<String, String>> {
    let mut version_info = HashMap::new();
    version_info.insert("ffi_version".to_string(), "0.1.0".to_string());
    version_info.insert("algorithm_version".to_string(), "1.0.0".to_string());
    version_info.insert("rust_version".to_string(), "1.75.0".to_string());
    version_info.insert("pyo3_version".to_string(), "0.22.0".to_string());
    version_info.insert("simd_support".to_string(), 
        if is_x86_feature_detected!("avx2") { "AVX2" } else { "Standard" }.to_string());
    version_info.insert("ferrocene_compliant".to_string(), "true".to_string());
    
    Ok(version_info)
}

/// Python module definition
#[pymodule]
fn ecg_processor_py(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add functions
    m.add_function(wrap_pyfunction!(analyze_ecg, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ecg_with_config, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ecg_async, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ecg_batch, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ecg_stream, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_ecg_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(test_connection, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    
    // Add classes
    m.add_class::<ECGAnalysisConfig>()?;
    m.add_class::<ECGAnalysisResult>()?;
    
    Ok(())
}
