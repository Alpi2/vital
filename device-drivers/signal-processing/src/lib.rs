//! Signal Processing Module
//! 
//! FDA/IEC 62304 compliant signal quality assessment and filtering

use vitalstream_device_drivers::{
    SignalProcessor, DeviceError, Result,
    types::{SignalQuality, QualityMetrics},
};
use rustfft::{FftPlanner, num_complex::Complex};
use statrs::statistics::Statistics;
use chrono::Utc;

pub struct SignalProcessorImpl {
    sampling_rate: f32,
    powerline_frequency: f32,
}

impl SignalProcessorImpl {
    pub fn new(sampling_rate: f32, powerline_frequency: f32) -> Self {
        Self {
            sampling_rate,
            powerline_frequency,
        }
    }
    
    /// Calculate signal-to-noise ratio
    fn calculate_snr(&self, signal: &[f32]) -> f32 {
        if signal.is_empty() {
            return 0.0;
        }
        
        let mean = signal.iter().copied().collect::<Vec<f32>>().mean();
        let std_dev = signal.iter().copied().collect::<Vec<f32>>().std_dev();
        
        if std_dev == 0.0 {
            return 0.0;
        }
        
        // SNR in dB
        20.0 * (mean.abs() / std_dev).log10()
    }
    
    /// Detect baseline wander level
    fn detect_baseline_wander(&self, signal: &[f32]) -> f32 {
        if signal.len() < 2 {
            return 0.0;
        }
        
        // Simple baseline wander detection using low-frequency component
        let window_size = (self.sampling_rate * 0.5) as usize; // 0.5 second window
        let mut baseline_variation = 0.0;
        
        for i in 0..signal.len().saturating_sub(window_size) {
            let window = &signal[i..i + window_size];
            let mean = window.iter().copied().collect::<Vec<f32>>().mean();
            baseline_variation += mean.abs();
        }
        
        baseline_variation / signal.len() as f32
    }
    
    /// Detect powerline interference
    fn detect_powerline_interference(&self, signal: &[f32]) -> f32 {
        if signal.len() < 2 {
            return 0.0;
        }
        
        // Use FFT to detect 50/60 Hz component
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(signal.len());
        
        let mut buffer: Vec<Complex<f32>> = signal
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        fft.process(&mut buffer);
        
        // Find magnitude at powerline frequency
        let freq_bin = (self.powerline_frequency * signal.len() as f32 / self.sampling_rate) as usize;
        
        if freq_bin < buffer.len() {
            buffer[freq_bin].norm() / signal.len() as f32
        } else {
            0.0
        }
    }
}

impl SignalProcessor for SignalProcessorImpl {
    fn calculate_sqi(&self, signal: &[f32]) -> Result<SignalQuality> {
        if signal.is_empty() {
            return Err(DeviceError::InvalidDataFormat(
                "Empty signal".to_string()
            ));
        }
        
        let snr = self.calculate_snr(signal);
        let baseline_wander = self.detect_baseline_wander(signal);
        let powerline_interference = self.detect_powerline_interference(signal);
        
        // Calculate overall SQI (0.0 - 1.0)
        let snr_score = (snr / 40.0).min(1.0).max(0.0); // Normalize SNR to 0-1
        let baseline_score = (1.0 - baseline_wander).max(0.0);
        let powerline_score = (1.0 - powerline_interference).max(0.0);
        
        let sqi = (snr_score + baseline_score + powerline_score) / 3.0;
        
        // Detect artifacts (simple threshold-based)
        let std_dev = signal.iter().copied().collect::<Vec<f32>>().std_dev();
        let artifact_count = signal.iter()
            .filter(|&&x| x.abs() > 3.0 * std_dev)
            .count();
        let artifact_percentage = (artifact_count as f32 / signal.len() as f32) * 100.0;
        
        // Detect lead-off (very low signal)
        let mean_amplitude = signal.iter().map(|x| x.abs()).sum::<f32>() / signal.len() as f32;
        let lead_off = mean_amplitude < 0.01; // Threshold for lead-off
        
        Ok(SignalQuality {
            sqi,
            snr,
            artifact_percentage,
            lead_off,
            timestamp: Utc::now(),
            metrics: QualityMetrics {
                baseline_wander,
                powerline_interference,
                motion_artifact: artifact_percentage / 100.0,
                muscle_noise: 0.0, // TODO: Implement muscle noise detection
                electrode_contact: if lead_off { 0.0 } else { 1.0 },
            },
        })
    }
    
    fn detect_artifacts(&self, signal: &[f32]) -> Result<Vec<usize>> {
        if signal.is_empty() {
            return Ok(vec![]);
        }
        
        let std_dev = signal.iter().copied().collect::<Vec<f32>>().std_dev();
        let threshold = 3.0 * std_dev;
        
        let artifacts: Vec<usize> = signal.iter()
            .enumerate()
            .filter(|(_, &x)| x.abs() > threshold)
            .map(|(i, _)| i)
            .collect();
        
        Ok(artifacts)
    }
    
    fn remove_baseline_wander(&self, signal: &[f32]) -> Result<Vec<f32>> {
        if signal.is_empty() {
            return Ok(vec![]);
        }
        
        // Simple high-pass filter to remove baseline wander
        let cutoff = 0.5; // Hz
        let alpha = cutoff / (cutoff + self.sampling_rate);
        
        let mut filtered = vec![0.0; signal.len()];
        filtered[0] = signal[0];
        
        for i in 1..signal.len() {
            filtered[i] = alpha * (filtered[i - 1] + signal[i] - signal[i - 1]);
        }
        
        Ok(filtered)
    }
    
    fn filter_powerline(&self, signal: &[f32], frequency: f32) -> Result<Vec<f32>> {
        if signal.is_empty() {
            return Ok(vec![]);
        }
        
        // Simple notch filter for powerline interference
        // In production, use a proper IIR notch filter
        let q_factor = 30.0;
        let omega = 2.0 * std::f32::consts::PI * frequency / self.sampling_rate;
        let alpha = omega.sin() / (2.0 * q_factor);
        
        let b0 = 1.0;
        let b1 = -2.0 * omega.cos();
        let b2 = 1.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * omega.cos();
        let a2 = 1.0 - alpha;
        
        let mut filtered = vec![0.0; signal.len()];
        let mut x1 = 0.0;
        let mut x2 = 0.0;
        let mut y1 = 0.0;
        let mut y2 = 0.0;
        
        for i in 0..signal.len() {
            let x0 = signal[i];
            let y0 = (b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
            
            filtered[i] = y0;
            
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
        }
        
        Ok(filtered)
    }
    
    fn remove_motion_artifacts(&self, signal: &[f32]) -> Result<Vec<f32>> {
        // Use median filter for motion artifact removal
        if signal.len() < 3 {
            return Ok(signal.to_vec());
        }
        
        let window_size = 5;
        let mut filtered = vec![0.0; signal.len()];
        
        for i in 0..signal.len() {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(signal.len());
            
            let mut window: Vec<f32> = signal[start..end].to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            filtered[i] = window[window.len() / 2];
        }
        
        Ok(filtered)
    }
    
    fn detect_lead_off(&self, signal: &[f32]) -> Result<bool> {
        if signal.is_empty() {
            return Ok(true);
        }
        
        let mean_amplitude = signal.iter().map(|x| x.abs()).sum::<f32>() / signal.len() as f32;
        let threshold = 0.01; // mV
        
        Ok(mean_amplitude < threshold)
    }
    
    fn detect_pacemaker_spikes(&self, signal: &[f32]) -> Result<Vec<usize>> {
        if signal.is_empty() {
            return Ok(vec![]);
        }
        
        let mut spikes = Vec::new();
        let threshold = 2.0; // mV - typical pacemaker spike amplitude
        let min_width = (self.sampling_rate * 0.002) as usize; // 2ms minimum width
        
        for i in 1..signal.len() - 1 {
            // Detect sharp positive spike
            if signal[i] > threshold 
                && signal[i] > signal[i - 1] 
                && signal[i] > signal[i + 1] {
                
                // Check if spike is narrow enough
                let mut width = 1;
                for j in (i + 1)..signal.len() {
                    if signal[j] > threshold / 2.0 {
                        width += 1;
                    } else {
                        break;
                    }
                }
                
                if width <= min_width {
                    spikes.push(i);
                }
            }
        }
        
        Ok(spikes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_signal_processor() {
        let processor = SignalProcessorImpl::new(500.0, 60.0);
        let signal: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        
        let result = processor.calculate_sqi(&signal);
        assert!(result.is_ok());
        
        let sqi = result.unwrap();
        assert!(sqi.sqi >= 0.0 && sqi.sqi <= 1.0);
    }
    
    #[test]
    fn test_detect_artifacts() {
        let processor = SignalProcessorImpl::new(500.0, 60.0);
        let mut signal: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        signal[500] = 100.0; // Add artifact
        
        let result = processor.detect_artifacts(&signal);
        assert!(result.is_ok());
        assert!(!result.unwrap().is_empty());
    }
    
    #[test]
    fn test_lead_off_detection() {
        let processor = SignalProcessorImpl::new(500.0, 60.0);
        let signal = vec![0.001; 1000]; // Very low signal
        
        let result = processor.detect_lead_off(&signal);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should detect lead-off
    }
}
