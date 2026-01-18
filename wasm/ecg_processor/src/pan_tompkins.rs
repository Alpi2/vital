use core::f64::consts::PI;
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

/// Butterworth bandpass filter (5-15 Hz, order 4) for QRS detection
pub struct BandpassFilter {
    b: Vec<f64>,  // Numerator coefficients
    a: Vec<f64>,  // Denominator coefficients
    x_hist: Vec<f64>,  // Input history
    y_hist: Vec<f64>,  // Output history
}

impl BandpassFilter {
    pub fn new(sampling_rate: f64) -> Self {
        // Design 4th order Butterworth bandpass filter (5-15 Hz)
        let low_cutoff = 5.0;
        let high_cutoff = 15.0;
        
        // Normalize frequencies
        let nyquist = sampling_rate / 2.0;
        let low_norm = low_cutoff / nyquist;
        let high_norm = high_cutoff / nyquist;
        
        // Calculate Butterworth coefficients using bilinear transform
        let (b, a) = Self::design_butterworth_bandpass(low_norm, high_norm);
        
        Self {
            b,
            a,
            x_hist: vec![0.0; 5],
            y_hist: vec![0.0; 5],
        }
    }
    
    fn design_butterworth_bandpass(low: f64, high: f64) -> (Vec<f64>, Vec<f64>) {
        // 4th order Butterworth bandpass filter design
        // Using pre-warping and bilinear transform
        
        // Pre-warp frequencies
        let w1 = (PI * low).tan();
        let w2 = (PI * high).tan();
        let bw = w2 - w1;
        let w0 = (w1 * w2).sqrt();
        
        // 2nd order sections coefficients
        let q = w0 / bw;
        let q2 = q * q;
        
        // First section (lowpass)
        let alpha1 = w0.sin() / (2.0 * q);
        let a1_0 = 1.0 + alpha1;
        let a1_1 = -2.0 * w0.cos();
        let a1_2 = 1.0 - alpha1;
        let b1_0 = (1.0 - w0.cos()) / 2.0;
        let b1_1 = 1.0 - w0.cos();
        let b1_2 = (1.0 - w0.cos()) / 2.0;
        
        // Second section (highpass)
        let alpha2 = w0.sin() / (2.0 * q);
        let a2_0 = 1.0 + alpha2;
        let a2_1 = -2.0 * w0.cos();
        let a2_2 = 1.0 - alpha2;
        let b2_0 = (1.0 + w0.cos()) / 2.0;
        let b2_1 = -(1.0 + w0.cos());
        let b2_2 = (1.0 + w0.cos()) / 2.0;
        
        // Combine sections (simplified for 4th order)
        let b = vec![
            b1_0 * b2_0 / a1_0 / a2_0,
            b1_1 * b2_0 / a1_0 / a2_0 + b1_0 * b2_1 / a1_0 / a2_0,
            b1_2 * b2_0 / a1_0 / a2_0 + b1_1 * b2_1 / a1_0 / a2_0 + b1_0 * b2_2 / a1_0 / a2_0,
            b1_2 * b2_1 / a1_0 / a2_0 + b1_1 * b2_2 / a1_0 / a2_0,
            b1_2 * b2_2 / a1_0 / a2_0,
        ];
        
        let a = vec![
            1.0,
            a1_1 / a1_0 + a2_1 / a2_0,
            a1_2 / a1_0 + a2_2 / a2_0 + a1_1 * a2_1 / a1_0 / a2_0,
            a1_2 * a2_1 / a1_0 / a2_0 + a1_1 * a2_2 / a1_0 / a2_0,
            a1_2 * a2_2 / a1_0 / a2_0,
        ];
        
        (b, a)
    }
    
    pub fn filter(&mut self, sample: f64) -> f64 {
        // Update input history
        self.x_hist.rotate_right(1);
        self.x_hist[0] = sample;
        
        // Apply IIR filter difference equation
        let mut y = 0.0;
        for i in 0..self.b.len() {
            y += self.b[i] * self.x_hist[i];
        }
        for i in 1..self.a.len() {
            y -= self.a[i] * self.y_hist[i];
        }
        y /= self.a[0];
        
        // Update output history
        self.y_hist.rotate_right(1);
        self.y_hist[0] = y;
        
        y
    }
}

/// 5-point derivative filter for QRS slope information
pub struct DerivativeFilter {
    history: Vec<f64>,
}

impl DerivativeFilter {
    pub fn new() -> Self {
        Self {
            history: vec![0.0; 5],
        }
    }
    
    pub fn filter(&mut self, sample: f64) -> f64 {
        self.history.rotate_right(1);
        self.history[0] = sample;
        
        // 5-point derivative: (1/8T)[-x(n-2) - 2x(n-1) + 2x(n+1) + x(n+2)]
        // Since we only have past samples, we use a causal approximation
        let derivative = (
            2.0 * self.history[0] 
            + self.history[1] 
            - self.history[3] 
            - 2.0 * self.history[4]
        ) / 8.0;
        
        derivative
    }
}

/// Moving window integrator (150ms window) for waveform integration
pub struct MovingWindowIntegrator {
    window_size: usize,
    window: Vec<f64>,
    sum: f64,
    index: usize,
}

impl MovingWindowIntegrator {
    pub fn new(sampling_rate: f64) -> Self {
        let window_size = (0.150 * sampling_rate) as usize;  // 150ms window
        Self {
            window_size,
            window: vec![0.0; window_size],
            sum: 0.0,
            index: 0,
        }
    }
    
    pub fn integrate(&mut self, sample: f64) -> f64 {
        // Square the sample
        let squared = sample * sample;
        
        // Remove oldest sample from sum
        self.sum -= self.window[self.index];
        
        // Add new sample
        self.window[self.index] = squared;
        self.sum += squared;
        
        // Update circular buffer index
        self.index = (self.index + 1) % self.window_size;
        
        // Return average
        self.sum / self.window_size as f64
    }
}

#[derive(Serialize, Deserialize)]
pub struct PanTompkinsResult {
    pub peaks: Vec<usize>,
    pub bpm: f64,
    pub rr_intervals: Vec<f64>,
    pub signal_quality: f64,
}

/// Pan-Tompkins QRS detection algorithm implementation
pub struct PanTompkins {
    sampling_rate: f64,
    bandpass: BandpassFilter,
    derivative: DerivativeFilter,
    integrator: MovingWindowIntegrator,
    
    // Adaptive thresholds
    spki: f64,  // Signal peak
    npki: f64,  // Noise peak
    threshold_i1: f64,
    threshold_i2: f64,
    
    // Detection state
    last_peak_time: usize,
    refractory_period: usize,
    rr_intervals: Vec<f64>,
    rr_average: f64,
    
    // Signal quality assessment
    noise_count: usize,
    signal_count: usize,
}

impl PanTompkins {
    pub fn new(sampling_rate: f64) -> Self {
        let refractory_period = (0.200 * sampling_rate) as usize;  // 200ms refractory
        
        Self {
            sampling_rate,
            bandpass: BandpassFilter::new(sampling_rate),
            derivative: DerivativeFilter::new(),
            integrator: MovingWindowIntegrator::new(sampling_rate),
            spki: 0.0,
            npki: 0.0,
            threshold_i1: 0.0,
            threshold_i2: 0.0,
            last_peak_time: 0,
            refractory_period,
            rr_intervals: Vec::new(),
            rr_average: 0.0,
            noise_count: 0,
            signal_count: 0,
        }
    }
    
    pub fn process(&mut self, samples: &[f64]) -> PanTompkinsResult {
        let mut peaks = Vec::new();
        let mut integrated_signal = Vec::new();
        
        // Initialize thresholds with first few samples
        let init_samples = samples.len().min(100);
        if init_samples > 0 {
            let mut max_val = 0.0;
            for &sample in samples.iter().take(init_samples) {
                let filtered = self.bandpass.filter(sample);
                let derivative = self.derivative.filter(filtered);
                let integrated = self.integrator.integrate(derivative);
                max_val = max_val.max(integrated);
            }
            self.spki = max_val * 0.25;
            self.npki = max_val * 0.125;
            self.threshold_i1 = self.npki + 0.25 * (self.spki - self.npki);
            self.threshold_i2 = 0.5 * self.threshold_i1;
        }
        
        // Process each sample through the pipeline
        for (i, &sample) in samples.iter().enumerate() {
            // 1. Bandpass filter
            let filtered = self.bandpass.filter(sample);
            
            // 2. Derivative
            let derivative = self.derivative.filter(filtered);
            
            // 3. Square and integrate
            let integrated = self.integrator.integrate(derivative);
            integrated_signal.push(integrated);
            
            // 4. Peak detection with adaptive thresholds
            if i > self.last_peak_time + self.refractory_period {
                if self.is_peak(&integrated_signal, i) {
                    peaks.push(i);
                    self.update_thresholds(integrated, true);
                    self.last_peak_time = i;
                    self.signal_count += 1;
                    
                    // Calculate RR interval
                    if peaks.len() > 1 {
                        let rr = (i - peaks[peaks.len() - 2]) as f64 / self.sampling_rate * 1000.0;
                        self.rr_intervals.push(rr);
                        
                        // Update RR average (last 8 intervals)
                        if self.rr_intervals.len() > 8 {
                            self.rr_intervals.remove(0);
                        }
                        self.rr_average = self.rr_intervals.iter().sum::<f64>() / self.rr_intervals.len() as f64;
                    }
                } else if integrated > self.threshold_i2 {
                    self.update_thresholds(integrated, false);
                    self.noise_count += 1;
                }
            }
        }
        
        // Searchback for missed beats
        self.searchback(&integrated_signal, &mut peaks);
        
        let bpm = self.calculate_bpm();
        let signal_quality = self.assess_signal_quality();
        
        PanTompkinsResult {
            peaks,
            bpm,
            rr_intervals: self.rr_intervals.clone(),
            signal_quality,
        }
    }
    
    fn is_peak(&self, signal: &[f64], index: usize) -> bool {
        if index < 2 || index >= signal.len() - 2 {
            return false;
        }
        
        let current = signal[index];
        
        // Check if local maximum
        if current <= signal[index - 1] || current <= signal[index + 1] {
            return false;
        }
        
        // Check against adaptive threshold
        current > self.threshold_i1
    }
    
    fn update_thresholds(&mut self, peak_value: f64, is_signal: bool) {
        if is_signal {
            // Update signal peak with exponential moving average
            self.spki = 0.125 * peak_value + 0.875 * self.spki;
        } else {
            // Update noise peak
            self.npki = 0.125 * peak_value + 0.875 * self.npki;
        }
        
        // Update adaptive thresholds
        self.threshold_i1 = self.npki + 0.25 * (self.spki - self.npki);
        self.threshold_i2 = 0.5 * self.threshold_i1;
    }
    
    fn searchback(&self, signal: &[f64], peaks: &mut Vec<usize>) {
        if self.rr_average < 0.36 || self.rr_average > 0.76 {
            return; // Skip searchback for irregular rhythms
        }
        
        let searchback_window = (self.rr_average * self.sampling_rate * 1.66) as usize;
        let mut i = self.last_peak_time;
        
        while i < signal.len() && i < self.last_peak_time + searchback_window {
            if i > self.last_peak_time + self.refractory_period {
                if self.is_searchback_peak(signal, i) {
                    peaks.push(i);
                    break;
                }
            }
            i += 1;
        }
    }
    
    fn is_searchback_peak(&self, signal: &[f64], index: usize) -> bool {
        if index < 2 || index >= signal.len() - 2 {
            return false;
        }
        
        let current = signal[index];
        
        // Check if local maximum
        if current <= signal[index - 1] || current <= signal[index + 1] {
            return false;
        }
        
        // Check against lower threshold for searchback
        current > self.threshold_i2
    }
    
    fn calculate_bpm(&self) -> f64 {
        if self.rr_intervals.is_empty() {
            return 0.0;
        }
        
        let avg_rr = self.rr_intervals.iter().sum::<f64>() / self.rr_intervals.len() as f64;
        60000.0 / avg_rr  // Convert ms to BPM
    }
    
    fn assess_signal_quality(&self) -> f64 {
        let total_samples = self.signal_count + self.noise_count;
        if total_samples == 0 {
            return 0.0;
        }
        
        let signal_ratio = self.signal_count as f64 / total_samples as f64;
        
        // Additional quality metrics can be added here
        // For now, use signal-to-noise ratio as quality indicator
        (signal_ratio * 0.7 + 0.3).min(1.0)
    }
}

// WASM bindings
#[wasm_bindgen]
pub struct PanTompkinsDetector {
    detector: PanTompkins,
}

#[wasm_bindgen]
impl PanTompkinsDetector {
    #[wasm_bindgen(constructor)]
    pub fn new(sampling_rate: f64) -> PanTompkinsDetector {
        PanTompkinsDetector {
            detector: PanTompkins::new(sampling_rate),
        }
    }
    
    #[wasm_bindgen]
    pub fn detect_peaks(&mut self, samples: &[f64]) -> JsValue {
        let result = self.detector.process(samples);
        serde_wasm_bindgen::to_value(&result).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn get_bpm(&self) -> f64 {
        self.detector.calculate_bpm()
    }
}

// Legacy functions for backward compatibility
#[wasm_bindgen]
pub fn bandpass_filter(input: &[f64], sampling_rate: f64, low_hz: f64, high_hz: f64, taps: usize) -> Vec<f64> {
    let mut filter = BandpassFilter::new(sampling_rate);
    input.iter().map(|&x| filter.filter(x)).collect()
}

#[wasm_bindgen]
pub fn analyze_ecg(input: &[f64], sampling_rate: f64) -> JsValue {
    let mut detector = PanTompkins::new(sampling_rate);
    let result = detector.process(input);
    serde_wasm_bindgen::to_value(&result).unwrap()
}
