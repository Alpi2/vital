pub mod pan_tompkins;

use core::f64::consts::PI;
use wasm_bindgen::prelude::*;
use serde::{Serialize};

// Re-export Pan-Tompkins functionality
pub use pan_tompkins::*;

#[derive(Serialize)]
pub struct ECGAnalysisResult {
    pub heart_rate: f64,
    pub rr_intervals: Vec<f64>,
    pub peaks: Vec<usize>,
    pub signal_quality: f64,
}

// Legacy function for backward compatibility
#[wasm_bindgen]
pub fn analyze_ecg(input: &[f64], sampling_rate: f64) -> JsValue {
    let mut detector = PanTompkins::new(sampling_rate);
    let result = detector.process(input);
    
    let legacy_result = ECGAnalysisResult {
        heart_rate: result.bpm,
        rr_intervals: result.rr_intervals,
        peaks: result.peaks,
        signal_quality: result.signal_quality,
    };
    
    serde_wasm_bindgen::to_value(&legacy_result).unwrap()
}

// Legacy bandpass filter function
#[wasm_bindgen]
pub fn bandpass_filter(input: &[f64], sampling_rate: f64, low_hz: f64, high_hz: f64, taps: usize) -> Vec<f64> {
    let mut filter = BandpassFilter::new(sampling_rate);
    input.iter().map(|&x| filter.filter(x)).collect()
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

fn hann(n: usize, m: usize) -> f64 {
    if m <= 1 {
        return 1.0;
    }
    0.5 - 0.5 * (2.0 * PI * (n as f64) / ((m - 1) as f64)).cos()
}

fn design_bandpass_fir(
    sampling_rate: f64,
    low_hz: f64,
    high_hz: f64,
    taps: usize,
    out: &mut [f64],
) {
    let m = taps;
    let fc1 = low_hz / sampling_rate;
    let fc2 = high_hz / sampling_rate;
    let mid = (m as isize - 1) / 2;

    for i in 0..m {
        let n = i as isize - mid;
        let x = n as f64;

        let h_lp2 = 2.0 * fc2 * sinc(2.0 * fc2 * x);
        let h_lp1 = 2.0 * fc1 * sinc(2.0 * fc1 * x);
        let h_bp = h_lp2 - h_lp1;

        out[i] = h_bp * hann(i, m);
    }

    let sum: f64 = out.iter().sum();
    if sum.abs() > 1e-12 {
        for v in out.iter_mut() {
            *v /= sum;
        }
    }
}
