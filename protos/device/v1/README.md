# Device Service Protocol Buffers

## Overview
The Device Service manages medical devices and IoT sensors in the VitalStream ecosystem.

## Messages

### Core Messages
- **Device** - Main device entity with full metadata
- **DeviceEvent** - Device events and state changes
- **DeviceDataPoint** - Real-time sensor data
- **DeviceCommand** - Commands sent to devices
- **DeviceConfiguration** - Device configuration settings
- **DeviceStatistics** - Aggregated device metrics

### Request/Response Pairs
- **CreateDeviceRequest/Response** - Register new devices
- **GetDevicesRequest/Response** - Query devices with filtering
- **GetDeviceRequest/Response** - Get specific device details
- **UpdateDeviceRequest/Response** - Update device information
- **DeleteDeviceRequest/Response** - Remove devices

### Data Streaming
- **StreamDeviceDataRequest** - Filter criteria for data streaming
- **StreamDeviceDataResponse** - Real-time data points
- **StreamDeviceEventsRequest** - Filter criteria for events
- **StreamDeviceEventsResponse** - Real-time device events

### Command Execution
- **SendDeviceCommandRequest/Response** - Send commands to devices
- **BatchDeviceRequest/Response** - Execute multiple operations

## Enums

### DeviceType
- `DEVICE_TYPE_UNSPECIFIED` (0)
- `DEVICE_TYPE_ECG_MONITOR` (1)
- `DEVICE_TYPE_BLOOD_PRESSURE` (2)
- `DEVICE_TYPE_PULSE_OXIMETER` (3)
- `DEVICE_TYPE_GLUCOMETER` (4)
- `DEVICE_TYPE_TEMPERATURE` (5)
- `DEVICE_TYPE_WEIGHT_SCALE` (6)
- `DEVICE_TYPE_ACTIVITY_TRACKER` (7)
- `DEVICE_TYPE_SLEEP_MONITOR` (8)
- `DEVICE_TYPE_MEDICATION_DISPENSER` (9)
- `DEVICE_TYPE_EMERGENCY_BUTTON` (10)
- `DEVICE_TYPE_FALL_DETECTOR` (11)
- `DEVICE_TYPE_GPS_TRACKER` (12)
- `DEVICE_TYPE_SMART_WATCH` (13)
- `DEVICE_TYPE_FITNESS_BAND` (14)
- `DEVICE_TYPE_HEARING_AID` (15)
- `DEVICE_TYPE_INSULIN_PUMP` (16)
- `DEVICE_TYPE_VENTILATOR` (17)
- `DEVICE_TYPE_INFUSION_PUMP` (18)
- `DEVICE_TYPE_HOSPITAL_BED` (19)
- `DEVICE_TYPE_WEARABLE_SENSOR` (20)

### DeviceStatus
- `DEVICE_STATUS_UNSPECIFIED` (0)
- `DEVICE_STATUS_OFFLINE` (1)
- `DEVICE_STATUS_ONLINE` (2)
- `DEVICE_STATUS_CONNECTING` (3)
- `DEVICE_STATUS_ERROR` (4)
- `DEVICE_STATUS_MAINTENANCE` (5)
- `DEVICE_STATUS_LOW_BATTERY` (6)
- `DEVICE_STATUS_CHARGING` (7)
- `DEVICE_STATUS_CALIBRATING` (8)
- `DEVICE_STATUS_UPDATING` (9)
- `DEVICE_STATUS_SUSPENDED` (10)

### ConnectionType
- `CONNECTION_TYPE_UNSPECIFIED` (0)
- `CONNECTION_TYPE_BLUETOOTH` (1)
- `CONNECTION_TYPE_WIFI` (2)
- `CONNECTION_TYPE_CELLULAR` (3)
- `CONNECTION_TYPE_ETHERNET` (4)
- `CONNECTION_TYPE_USB` (5)
- `CONNECTION_TYPE_SERIAL` (6)
- `CONNECTION_TYPE_ZIGBEE` (7)
- `CONNECTION_TYPE_LORA` (8)
- `CONNECTION_TYPE_NFC` (9)

### DeviceEvent.EventType
- `EVENT_TYPE_UNSPECIFIED` (0)
- `EVENT_TYPE_CONNECTED` (1)
- `EVENT_TYPE_DISCONNECTED` (2)
- `EVENT_TYPE_LOW_BATTERY` (3)
- `EVENT_TYPE_ERROR` (4)
- `EVENT_TYPE_MAINTENANCE_REQUIRED` (5)
- `EVENT_TYPE_FIRMWARE_UPDATE` (6)
- `EVENT_TYPE_CONFIGURATION_CHANGED` (7)
- `EVENT_TYPE_LOCATION_CHANGED` (8)
- `EVENT_TYPE_SIGNAL_LOST` (9)
- `EVENT_TYPE_CALIBRATION_REQUIRED` (10)
- `EVENT_TYPE_TAMPER_DETECTED` (11)

## Service RPCs

### Device Management
- `CreateDevice` - Register new device
- `GetDevices` - Query devices with pagination
- `GetDevice` - Get specific device details
- `UpdateDevice` - Update device information
- `DeleteDevice` - Remove device from system

### Data Streaming
- `StreamDeviceData` - Real-time sensor data streaming
- `StreamDeviceEvents` - Real-time event streaming

### Device Control
- `SendDeviceCommand` - Send commands to devices
- `DeviceChannel` - Bidirectional device communication

### Analytics
- `GetDeviceStatistics` - Get device usage statistics
- `BatchDeviceOperations` - Execute batch operations
- `HealthCheck` - Service health check

## Usage Examples

### Register Device
```python
request = CreateDeviceRequest(
    patient_id="patient-123",
    serial_number="ECG-001",
    model="VitalECG-Pro",
    manufacturer="VitalStream",
    type=DeviceType.DEVICE_TYPE_ECG_MONITOR,
    connection_type=ConnectionType.CONNECTION_TYPE_BLUETOOTH
)
```

### Stream Device Data
```python
request = StreamDeviceDataRequest(
    device_id="device-123",
    data_types=["ecg_signal", "heart_rate"],
    real_time=True
)
```

### Send Command
```python
request = SendDeviceCommandRequest(
    device_id="device-123",
    command="start_monitoring",
    parameters={"sampling_rate": 250, "duration": 300}
)
```

## Best Practices

1. **Device Registration**
   - Use unique serial numbers
   - Include manufacturer and model information
   - Set appropriate device type

2. **Data Streaming**
   - Use appropriate sampling rates
   - Implement buffering for reliability
   - Handle connection failures gracefully

3. **Command Execution**
   - Include timeout values
   - Validate parameters before sending
   - Handle command failures appropriately

4. **Error Handling**
   - Implement retry logic for transient failures
   - Log all device interactions
   - Monitor device health continuously
