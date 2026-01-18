#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandpass_filter_creation() {
        let filter = BandpassFilter::new(250.0);
        assert_eq!(filter.b.len(), 5);
        assert_eq!(filter.a.len(), 5);
        assert_eq!(filter.x_hist.len(), 5);
        assert_eq!(filter.y_hist.len(), 5);
    }

    #[test]
    fn test_bandpass_filter_processing() {
        let mut filter = BandpassFilter::new(250.0);
        let samples = vec![0.0; 100];
        let filtered: Vec<f64> = samples.iter()
            .map(|&s| filter.filter(s))
            .collect();
        assert_eq!(filtered.len(), 100);
    }

    #[test]
    fn test_derivative_filter() {
        let mut filter = DerivativeFilter::new();
        assert_eq!(filter.history.len(), 5);
        
        // Test with step input
        let output = filter.filter(1.0);
        assert!(output != 0.0);
    }

    #[test]
    fn test_moving_window_integrator() {
        let mut integrator = MovingWindowIntegrator::new(250.0);
        let window_size = (0.150 * 250.0) as usize;
        assert_eq!(integrator.window_size, window_size);
        assert_eq!(integrator.window.len(), window_size);
        
        let output = integrator.integrate(2.0);
        assert!(output >= 0.0);
    }

    #[test]
    fn test_pan_tompkins_creation() {
        let pt = PanTompkins::new(250.0);
        assert_eq!(pt.sampling_rate, 250.0);
        assert_eq!(pt.refractory_period, (0.200 * 250.0) as usize);
    }

    #[test]
    fn test_pan_tompkins_synthetic_signal() {
        let mut pt = PanTompkins::new(250.0);
        
        // Generate synthetic ECG with known peaks
        let mut samples = vec![0.0; 1000];
        samples[250] = 1.0;  // Peak at 1 second
        samples[500] = 1.0;  // Peak at 2 seconds
        samples[750] = 1.0;  // Peak at 3 seconds
        
        let result = pt.process(&samples);
        
        // Should detect approximately 3 peaks
        assert!(result.peaks.len() >= 2);
        assert!(result.bpm > 40.0 && result.bpm < 200.0);
    }

    #[test]
    fn test_rr_interval_calculation() {
        let mut pt = PanTompkins::new(250.0);
        
        // Create signal with peaks 1 second apart
        let mut samples = vec![0.0; 750];
        samples[250] = 1.0;  // 1 second
        samples[500] = 1.0;  // 2 seconds
        
        let result = pt.process(&samples);
        
        if result.rr_intervals.len() > 0 {
            // RR interval should be approximately 1000ms (1 second)
            let rr = result.rr_intervals[0];
            assert!(rr > 900.0 && rr < 1100.0);
        }
    }

    #[test]
    fn test_bpm_calculation() {
        let mut pt = PanTompkins::new(250.0);
        pt.rr_intervals = vec![833.0, 833.0, 833.0];  // 72 BPM
        let bpm = pt.calculate_bpm();
        assert!((bpm - 72.0).abs() < 1.0);
    }

    #[test]
    fn test_signal_quality_assessment() {
        let mut pt = PanTompkins::new(250.0);
        pt.signal_count = 80;
        pt.noise_count = 20;
        
        let quality = pt.assess_signal_quality();
        assert!(quality > 0.7);
        assert!(quality <= 1.0);
    }

    #[test]
    fn test_adaptive_thresholds() {
        let mut pt = PanTompkins::new(250.0);
        
        // Initialize thresholds
        pt.spki = 1.0;
        pt.npki = 0.1;
        pt.threshold_i1 = pt.npki + 0.25 * (pt.spki - pt.npki);
        pt.threshold_i2 = 0.5 * pt.threshold_i1;
        
        let initial_threshold_i1 = pt.threshold_i1;
        
        // Update with signal peak
        pt.update_thresholds(2.0, true);
        
        // Threshold should increase
        assert!(pt.threshold_i1 > initial_threshold_i1);
        assert!(pt.spki > 1.0);
    }

    #[test]
    fn test_wasm_binding() {
        let mut detector = PanTompkinsDetector::new(250.0);
        let samples = vec![0.0; 100];
        
        let result = detector.detect_peaks(&samples);
        assert!(result.is_object());
        
        let bpm = detector.get_bpm();
        assert!(bpm >= 0.0);
    }

    #[test]
    fn test_empty_signal() {
        let mut pt = PanTompkins::new(250.0);
        let result = pt.process(&[]);
        
        assert_eq!(result.peaks.len(), 0);
        assert_eq!(result.bpm, 0.0);
        assert_eq!(result.rr_intervals.len(), 0);
    }

    #[test]
    fn test_high_frequency_noise() {
        let mut pt = PanTompkins::new(250.0);
        
        // Generate high frequency noise
        let samples: Vec<f64> = (0..1000)
            .map(|i| (i as f64 * 0.1).sin() * 0.1)
            .collect();
        
        let result = pt.process(&samples);
        
        // Should detect very few or no peaks in pure noise
        assert!(result.peaks.len() <= 2);
        assert!(result.signal_quality < 0.5);
    }

    #[test]
    fn test_regular_rhythm() {
        let mut pt = PanTompkins::new(360.0);  // Standard clinical sampling rate
        
        // Generate regular rhythm at 60 BPM (1 second intervals)
        let mut samples = vec![0.0; 3600];  // 10 seconds
        for i in 0..10 {
            samples[i * 360] = 1.0;  // Peak every 1 second
        }
        
        let result = pt.process(&samples);
        
        // Should detect peaks around 60 BPM
        assert!(result.bpm > 50.0 && result.bpm < 70.0);
        assert!(result.peaks.len() >= 8);
        
        // RR intervals should be consistent
        if result.rr_intervals.len() > 1 {
            let rr_std = calculate_std(&result.rr_intervals);
            assert!(rr_std < 100.0);  // Low variability in regular rhythm
        }
    }
}

fn calculate_std(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    
    variance.sqrt()
}
