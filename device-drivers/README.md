# VitalStream Device Drivers

FDA/IEC 62304 compliant medical device driver library written in Rust.

## Overview

This library provides safe, deterministic interfaces for medical device communication, supporting multiple connection types and protocols.

## Features

### Connection Types
- **BLE (Bluetooth Low Energy)**: Wireless medical device connectivity
- **USB**: High-speed wired connections
- **Serial (RS-232)**: Legacy device support
- **Wi-Fi**: Network-based device communication

### Protocols
- **HL7 v2.x**: Healthcare messaging standard
- **DICOM**: Medical imaging and waveform storage
- **IEEE 11073**: Personal Health Device communication

### Signal Processing
- Signal Quality Index (SQI) calculation
- Artifact detection and removal
- Baseline wander correction
- Powerline interference filtering (50/60 Hz)
- Motion artifact removal
- Lead-off detection
- Pacemaker spike detection

## Architecture

```
device-drivers/
├── src/
│   ├── lib.rs              # Main library
│   ├── error.rs            # Error types
│   ├── traits.rs           # Core traits
│   ├── types.rs            # Common types
│   └── protocols/          # Protocol implementations
│       ├── hl7.rs
│       ├── dicom.rs
│       └── ieee11073.rs
├── ble-driver/             # BLE driver
├── usb-driver/             # USB driver
├── serial-driver/          # Serial port driver
└── signal-processing/      # Signal processing algorithms
```

## Usage

### Basic Example

```rust
use vitalstream_device_drivers::*;
use ble_driver::BLEDriver;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize library
    vitalstream_device_drivers::init()?;
    
    // Create BLE driver
    let mut driver = BLEDriver::new().await?;
    
    // Discover devices
    let devices = driver.discover().await?;
    
    // Connect to device
    driver.connect("DEVICE_ID").await?;
    
    // Read data
    let mut buffer = vec![0u8; 1024];
    let bytes_read = driver.receive(&mut buffer).await?;
    
    // Disconnect
    driver.disconnect().await?;
    
    Ok(())
}
```

### Signal Processing Example

```rust
use signal_processing::SignalProcessorImpl;
use vitalstream_device_drivers::SignalProcessor;

fn main() -> Result<()> {
    let processor = SignalProcessorImpl::new(500.0, 60.0);
    
    // Sample ECG signal
    let signal: Vec<f32> = vec![/* ... */];
    
    // Calculate signal quality
    let quality = processor.calculate_sqi(&signal)?;
    println!("SQI: {}, SNR: {} dB", quality.sqi, quality.snr);
    
    // Remove baseline wander
    let filtered = processor.remove_baseline_wander(&signal)?;
    
    // Detect artifacts
    let artifacts = processor.detect_artifacts(&signal)?;
    
    Ok(())
}
```

## Safety and Compliance

### FDA/IEC 62304 Compliance
- Memory-safe Rust implementation
- Comprehensive error handling
- Deterministic performance
- Extensive testing coverage
- Audit logging

### Safety Features
- Overflow checks enabled in release builds
- No unsafe code in critical paths
- Panic-free operation (abort on panic)
- Link-time optimization (LTO)

## Building

```bash
# Build all workspace members
cargo build --release

# Run tests
cargo test

# Run with logging
RUST_LOG=info cargo run
```

## Testing

```bash
# Unit tests
cargo test

# Integration tests
cargo test --test '*'

# With coverage
cargo tarpaulin --out Html
```

## Dependencies

- **tokio**: Async runtime
- **btleplug**: BLE support
- **rusb**: USB support
- **tokio-serial**: Serial port support
- **rustfft**: FFT for signal processing
- **statrs**: Statistical functions

## Integration with VitalStream

This library integrates with:
- **Backend (Python)**: Via FFI or REST API
- **Java HL7/DICOM Service**: For enterprise integration
- **Frontend (Angular)**: Via WebSocket for real-time data

## Roadmap

### Phase 2.1: Device Integration
- [x] BLE driver implementation
- [x] USB driver implementation
- [x] Serial driver implementation
- [x] Basic protocol support
- [ ] Wi-Fi driver implementation
- [ ] Specific device drivers (Philips, GE, Mindray)

### Phase 2.2: Signal Processing
- [x] SQI calculation
- [x] Artifact detection
- [x] Baseline wander removal
- [x] Powerline filtering
- [x] Lead-off detection
- [x] Pacemaker spike detection
- [ ] Advanced wavelet analysis
- [ ] Machine learning-based artifact removal

### Phase 2.3: Protocol Enhancement
- [x] Basic HL7 support
- [x] Basic DICOM support
- [x] IEEE 11073 support
- [ ] Full HL7 FHIR integration (via Java service)
- [ ] DICOM C-STORE/C-FIND operations
- [ ] Real-time streaming protocols

## License

MIT License - See LICENSE file for details

## Contributing

See CONTRIBUTING.md for guidelines.

## Support

For issues and questions, please open a GitHub issue.
