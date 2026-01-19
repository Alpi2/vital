# ECG Device Adapter Pattern Documentation

## Overview

The ECG Device Adapter Pattern provides a unified interface for communicating with various ECG devices through a consistent API. This implementation supports both mock devices for testing and real hardware devices through Rust FFI integration, with features like health monitoring, automatic reconnection, and device pooling.

## Architecture

### Core Components

1. **DeviceInterface** - Abstract base class defining the contract for all ECG devices
2. **MockECGDevice** - Synthetic ECG device for testing and development
3. **RealECGDevice** - Real hardware device interface with Rust FFI
4. **DeviceFactory** - Factory for creating device instances based on configuration
5. **DeviceHealthMonitor** - Continuous health monitoring with automatic reconnection
6. **ConnectionManager** - Multi-device connection management
7. **DevicePool** - Device pooling with load balancing strategies

### Design Patterns

- **Adapter Pattern**: Provides consistent interface for different device types
- **Factory Pattern**: Creates device instances based on configuration
- **Observer Pattern**: Event-driven status and error notifications
- **Pool Pattern**: Resource pooling for efficient device management

## Device Types

### Mock Device
- **Purpose**: Testing, development, and demonstration
- **Features**: Realistic ECG waveform generation, anomaly injection, noise simulation
- **Configuration**: Heart rate, sampling rate, noise level, signal quality

### Real Devices
- **USB**: Direct USB connection to ECG hardware
- **BLE**: Bluetooth Low Energy wireless devices
- **Serial**: RS232/RS485 serial port devices
- **Network**: TCP/IP network-connected devices

## Configuration

### Basic Device Configuration

```python
# Mock Device Configuration
mock_config = {
    "use_mock": True,
    "heart_rate": 72,
    "sampling_rate": 250,
    "noise_level": 0.05,
    "signal_quality": 0.95,
    "num_leads": 1
}

# USB Device Configuration
usb_config = {
    "device_type": "usb",
    "connection_params": {
        "vendor_id": "0x1234",
        "product_id": "0x5678",
        "interface": 0
    },
    "rust_lib_path": "device-drivers/target/release/libdevice_drivers.so"
}

# BLE Device Configuration
ble_config = {
    "device_type": "ble",
    "connection_params": {
        "mac_address": "00:00:00:00:00:00",
        "service_uuid": "0000180d-0000-1000-8000-00805f9b34fb"
    },
    "rust_lib_path": "device-drivers/target/release/libdevice_drivers.so"
}
```

### Health Monitoring Configuration

```python
health_config = {
    "check_interval": 5.0,  # seconds
    "thresholds": {
        "min_signal_quality": 0.7,
        "max_latency_ms": 100.0,
        "min_throughput_bps": 1000.0,
        "max_error_rate": 0.05,
        "battery_warning": 20.0,
        "battery_critical": 10.0,
        "temperature_warning": 40.0,
        "temperature_critical": 50.0
    },
    "reconnection": {
        "enabled": True,
        "max_attempts": 5,
        "initial_delay": 1.0,
        "max_delay": 60.0,
        "backoff_multiplier": 2.0,
        "jitter": True
    }
}
```

### Device Pool Configuration

```python
pool_config = {
    "max_devices": 10,
    "min_devices": 1,
    "strategy": "round_robin",  # round_robin, least_loaded, best_quality, random, priority
    "health_check_interval": 30.0,
    "device_config": {
        "use_mock": True,
        "sampling_rate": 250
    }
}
```

## Usage Examples

### Basic Device Usage

```python
from app.devices import DeviceFactory

# Create device factory
factory = DeviceFactory()

# Create mock device
device = factory.create_device("ecg-001", {
    "use_mock": True,
    "heart_rate": 75,
    "sampling_rate": 250
})

# Connect to device
if device.connect():
    print(f"Connected to {device.device_id}")
    
    # Read ECG samples
    samples = device.read_samples(250)  # 1 second at 250Hz
    print(f"Read {len(samples)} samples")
    
    # Get device information
    info = device.get_info()
    print(f"Device: {info.manufacturer} {info.model}")
    
    # Get health status
    health = device.get_health()
    print(f"Signal quality: {health.signal_quality:.2f}")
    
    # Disconnect
    device.disconnect()
```

### Connection Manager Usage

```python
from app.devices import ConnectionManager, ConnectionEvent

# Create connection manager
manager = ConnectionManager(max_connections=5)

# Add event callback
def on_device_connected(event):
    print(f"Device {event.device_id} connected")

manager.add_event_callback(ConnectionEvent.DEVICE_CONNECTED, on_device_connected)

# Connect multiple devices
devices = [
    ("ecg-001", {"use_mock": True, "heart_rate": 60}),
    ("ecg-002", {"use_mock": True, "heart_rate": 80}),
    ("ecg-003", {"use_mock": True, "heart_rate": 100})
]

for device_id, config in devices:
    manager.connect_device(device_id, config)

# Read from specific device
samples = manager.read_samples("ecg-001", 100)

# Get device health
health = manager.get_device_health("ecg-001")

# Disconnect all devices
manager.disconnect_all()
```

### Streaming Data

```python
# Streaming with individual device
device = factory.create_device("stream-001", {"use_mock": True})
device.connect()

def data_callback(samples):
    # Process ECG samples
    print(f"Received {len(samples)} samples")
    # Add your processing logic here

# Start streaming
device.start_streaming(data_callback)

# Let it run for a while
import time
time.sleep(10)

# Stop streaming
device.stop_streaming()
device.disconnect()

# Streaming with connection manager
manager = ConnectionManager()
manager.connect_device("stream-002", {"use_mock": True})

def stream_callback(samples):
    # Handle streaming data
    pass

manager.start_streaming("stream-002", stream_callback)
```

### Health Monitoring

```python
from app.devices import DeviceHealthMonitor, HealthCheckLevel

device = factory.create_device("health-001", {"use_mock": True})
device.connect()

# Configure health monitoring
health_config = {
    "check_interval": 2.0,
    "thresholds": {
        "min_signal_quality": 0.8,
        "max_latency_ms": 50.0
    }
}

# Create health monitor
monitor = DeviceHealthMonitor(device, health_config)

def health_callback(check):
    if check.level == HealthCheckLevel.WARNING:
        print(f"Health warning: {check.message}")
    elif check.level == HealthCheckLevel.ERROR:
        print(f"Health error: {check.message}")

def reconnection_callback(success):
    if success:
        print("Device reconnected successfully")
    else:
        print("Reconnection failed")

monitor.add_health_callback(health_callback)
monitor.add_reconnection_callback(reconnection_callback)

# Start monitoring
monitor.start_monitoring()

# Let it run
time.sleep(30)

# Stop monitoring
monitor.stop_monitoring()
device.disconnect()
```

### Device Pooling

```python
from app.devices.device_pool import DevicePool, PoolStrategy

# Create device pool
pool_config = {
    "max_devices": 5,
    "min_devices": 2,
    "strategy": "least_loaded",
    "device_config": {
        "use_mock": True,
        "sampling_rate": 250
    }
}

pool = DevicePool("ecg-pool", pool_config)

# Acquire device from pool
device = pool.acquire_device(timeout=5.0)
if device:
    try:
        # Use device
        samples = device.read_samples(100)
        print(f"Read {len(samples)} samples from pooled device")
    finally:
        # Release device back to pool
        pool.release_device(device)

# Get pool status
status = pool.get_pool_status()
print(f"Pool utilization: {status['utilization']:.2%}")

# Scale pool
pool.scale_pool(3)

# Shutdown pool
pool.shutdown()
```

## Mock Device Features

### Synthetic ECG Generation

The mock device generates realistic ECG waveforms with:

- **P wave**: Atrial depolarization
- **QRS complex**: Ventricular depolarization
- **T wave**: Ventricular repolarization
- **Baseline noise**: Realistic signal noise
- **Heart rate variability**: Natural HRV patterns

### Anomaly Injection

```python
from app.devices.mock_device import AnomalyType

device = factory.create_device("anomaly-test", {"use_mock": True})
device.connect()

# Inject different anomalies
anomalies = [
    AnomalyType.AFIB,  # Atrial fibrillation
    AnomalyType.PVC,   # Premature ventricular contraction
    AnomalyType.ST_ELEVATION,  # ST segment elevation
    AnomalyType.ST_DEPRESSION,  # ST segment depression
    AnomalyType.VENTRICULAR_TACHYCARDIA,
    AnomalyType.HEART_BLOCK,
    AnomalyType.PREMATURE_ATRIAL
]

for anomaly in anomalies:
    device.inject_anomaly(anomaly, duration=2.0)  # 2 seconds
    samples = device.read_samples(500)  # 2 seconds at 250Hz
    # Process anomalous ECG data
```

### Configuration Options

```python
# Advanced mock device configuration
advanced_config = {
    "use_mock": True,
    "heart_rate": 72,
    "sampling_rate": 250,
    "noise_level": 0.05,  # 0.0 to 1.0
    "signal_quality": 0.95,  # 0.0 to 1.0
    "num_leads": 1,
    "buffer_size": 10000,
    "anomaly_probability": 0.01,  # 1% chance of random anomaly
    "hrv_enabled": True,  # Heart rate variability
    "baseline_wander": 0.02  # Baseline wander amplitude
}
```

## Real Device Integration

### Rust FFI Setup

Real devices communicate through Rust drivers using FFI:

1. **Build Rust library**:
   ```bash
   cd device-drivers
   cargo build --release
   ```

2. **Configure Python**:
   ```python
   real_config = {
       "device_type": "usb",
       "rust_lib_path": "device-drivers/target/release/libdevice_drivers.so",
       "connection_params": {
           "vendor_id": "0x1234",
           "product_id": "0x5678"
       }
   }
   ```

3. **Use device**:
   ```python
   device = factory.create_device("real-001", real_config)
   device.connect()
   samples = device.read_samples(250)
   ```

### Supported Device Types

#### USB Devices
- Direct USB connection
- Vendor/product ID identification
- Multiple interface support

#### BLE Devices
- Bluetooth Low Energy
- Service discovery
- Characteristic read/write

#### Serial Devices
- RS232/RS485 communication
- Configurable baud rates
- Hardware flow control

#### Network Devices
- TCP/IP connectivity
- WebSocket support
- Protocol negotiation

## Error Handling

### Exception Types

```python
from app.devices.interface import (
    DeviceError, DeviceConnectionError, DeviceReadError,
    DeviceTimeoutError, DeviceConfigurationError
)

try:
    device = factory.create_device("test-001", config)
    device.connect()
    samples = device.read_samples(100)
except DeviceConnectionError as e:
    print(f"Connection failed: {e}")
except DeviceReadError as e:
    print(f"Read failed: {e}")
except DeviceTimeoutError as e:
    print(f"Operation timed out: {e}")
except DeviceConfigurationError as e:
    print(f"Configuration error: {e}")
```

### Error Recovery

The system provides automatic error recovery:

1. **Connection Loss**: Automatic reconnection with exponential backoff
2. **Read Failures**: Retry with timeout escalation
3. **Health Issues**: Device isolation and recovery attempts
4. **Configuration Errors**: Validation and fallback to defaults

## Performance Considerations

### Memory Management

- **Sample Buffers**: Configurable buffer sizes with automatic cleanup
- **Device Pools**: Resource pooling to minimize allocation overhead
- **Streaming**: Efficient data transfer with minimal copying

### Threading

- **Thread Safety**: All operations are thread-safe
- **Lock Management**: Minimal lock contention with read-write locks
- **Async Operations**: Non-blocking I/O where supported

### Optimization Tips

```python
# Use appropriate buffer sizes
config = {
    "buffer_size": 10000,  # Balance memory vs. performance
    "sampling_rate": 250   # Match device capabilities
}

# Enable health monitoring only when needed
health_config = {
    "check_interval": 10.0,  # Longer intervals for less overhead
    "reconnection": {
        "enabled": False  # Disable if not needed
    }
}

# Use device pooling for high-throughput scenarios
pool_config = {
    "strategy": "least_loaded",  # Optimize for load distribution
    "max_devices": 10  # Scale based on expected load
}
```

## Testing

### Unit Tests

```bash
# Run all device tests
pytest backend/tests/devices/ -v

# Run specific test suites
pytest backend/tests/devices/test_interface.py -v
pytest backend/tests/devices/test_mock_device.py -v
pytest backend/tests/devices/test_integration.py -v
```

### Test Coverage

The test suite covers:
- Device interface contracts
- Mock device functionality
- Factory pattern implementation
- Connection management
- Health monitoring
- Error handling
- Thread safety
- Performance characteristics

### Mock Testing

For testing without hardware:

```python
# Enable mock mode in configuration
settings.use_mock_devices = True

# Or use mock configuration explicitly
mock_config = {"use_mock": True, "sampling_rate": 250}
device = factory.create_device("test-001", mock_config)
```

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check device configuration
   - Verify hardware connections
   - Review driver installation

2. **Performance Issues**
   - Adjust buffer sizes
   - Optimize sampling rates
   - Enable device pooling

3. **Memory Leaks**
   - Ensure proper device cleanup
   - Monitor buffer usage
   - Check thread termination

4. **Health Monitoring**
   - Adjust threshold values
   - Review reconnection settings
   - Check network connectivity

### Debug Logging

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Device-specific logging
logger = logging.getLogger("app.devices")
logger.setLevel(logging.DEBUG)
```

### Performance Monitoring

Monitor device performance:

```python
# Get device metrics
metrics = device.get_metrics()
print(f"Average read time: {metrics.average_read_time*1000:.2f}ms")
print(f"Error rate: {metrics.errors_count/metrics.samples_received:.2%}")

# Get pool statistics
status = pool.get_pool_status()
print(f"Pool utilization: {status['utilization']:.2%}")
print(f"Average load: {status['average_load']:.2f}")
```

## Future Enhancements

### Planned Features

1. **Advanced Anomaly Detection**
   - Machine learning-based anomaly detection
   - Real-time arrhythmia classification
   - Predictive health monitoring

2. **Device Discovery**
   - Automatic device detection
   - Network scanning for BLE devices
   - USB device enumeration

3. **Configuration Management**
   - Dynamic configuration updates
   - Configuration templates
   - Device profile management

4. **Performance Optimization**
   - GPU acceleration for signal processing
   - Zero-copy data transfer
   - Adaptive buffer management

### Extension Points

The architecture supports extensions through:

- **Custom Device Types**: Implement DeviceInterface for new devices
- **Health Check Plugins**: Add custom health monitoring
- **Load Balancing Strategies**: Implement new pool strategies
- **Event Handlers**: Add custom event processing

## API Reference

### DeviceInterface Methods

```python
# Connection management
connect() -> bool
disconnect() -> bool
is_connected() -> bool

# Data operations
read_samples(count: int, timeout: Optional[float] = None) -> List[float]
start_streaming(callback: Callable[[List[float]], None]) -> bool
stop_streaming() -> bool

# Device information
get_info() -> DeviceInfo
get_health() -> DeviceHealth
get_metrics() -> DeviceMetrics

# Configuration
calibrate() -> bool
get_lead_status() -> Dict[str, Any]
```

### ConnectionManager Methods

```python
# Device management
connect_device(device_id: str, config: Dict[str, Any]) -> bool
disconnect_device(device_id: str) -> bool
get_device(device_id: str) -> Optional[DeviceInterface]
get_connected_devices() -> List[str]

# Data operations
read_samples(device_id: str, count: int, timeout: Optional[float] = None) -> List[float]
start_streaming(device_id: str, callback: Callable[[List[float]], None]) -> bool
stop_streaming(device_id: str) -> bool

# Monitoring
get_device_health(device_id: str) -> Optional[Dict[str, Any]]
get_connection_stats(device_id: Optional[str] = None) -> Dict[str, Any]

# Events
add_event_callback(event_type: ConnectionEvent, callback: Callable) -> None
get_events(timeout: Optional[float] = None) -> List[ConnectionEventInfo]
```

### DeviceFactory Methods

```python
# Device creation
create_device(device_id: str, config: Optional[Dict[str, Any]] = None) -> DeviceInterface
get_available_devices() -> List[Dict[str, Any]]
create_device_from_info(device_info: Dict[str, Any]) -> DeviceInterface

# Configuration
save_configuration(device_id: str, config: Dict[str, Any], filename: Optional[str] = None) -> None
load_configuration(device_id: str, filename: Optional[str] = None) -> Optional[Dict[str, Any]]
```

This documentation provides comprehensive guidance for using the ECG Device Adapter Pattern in various scenarios, from basic device communication to advanced multi-device management with health monitoring and automatic recovery.
