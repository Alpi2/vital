# VitalStream ML Inference Engine

## Production Inference - Rust Implementation

### Özellikler

- **Düşük Latency**: <10ms inference süresi
- **Deterministik**: Tutarlı performans
- **Memory Safe**: Rust'un güvenlik garantileri
- **FDA/IEC 62304 Uyumlu**: Tıbbi cihaz standartlarına uygun
- **ONNX Desteği**: PyTorch/TensorFlow modellerini çalıştırma

### Mimari

```
ml-inference/
├── src/
│   ├── lib.rs              # Library root
│   ├── models/             # Model wrappers
│   │   ├── arrhythmia.rs   # Arrhythmia detection
│   │   ├── mi_detection.rs # MI detection
│   │   └── hrv_analysis.rs # HRV analysis
│   ├── preprocessing/     # Signal preprocessing
│   ├── clinical/          # Clinical decision support
│   │   ├── ews.rs          # Early Warning Score
│   │   ├── sepsis.rs       # Sepsis detection
│   │   └── cardiac.rs      # Cardiac arrest risk
│   └── utils/             # Utilities
├── models/                # ONNX model files
├── benches/               # Performance benchmarks
└── tests/                 # Integration tests
```

### Kullanım

```rust
use vital_ml_inference::models::ArrhythmiaDetector;

// Load model
let detector = ArrhythmiaDetector::new("models/arrhythmia.onnx")?;

// Preprocess ECG signal
let signal = preprocess_ecg(&raw_signal)?;

// Run inference
let prediction = detector.predict(&signal)?;

println!("Arrhythmia: {:?}, Confidence: {:.2}%", 
         prediction.class, prediction.confidence * 100.0);
```

### Performans

- **Inference Time**: 5-8ms (single ECG window)
- **Throughput**: 125+ inferences/second
- **Memory**: <100MB
- **CPU Usage**: <10% (single core)

### Regülasyon Uyumu

- IEC 62304 Class C (highest safety)
- ISO 14971 Risk Management
- Deterministic execution
- Comprehensive logging
- Validation test suite

### Test

```bash
# Unit tests
cargo test

# Benchmarks
cargo bench

# Coverage
cargo tarpaulin --out Html
```
