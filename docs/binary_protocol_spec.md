# Binary Protocol Specification

## Overview

This document describes the binary WebSocket protocol used for high-performance communication between the ECG monitoring frontend and backend. The protocol uses MessagePack serialization with custom binary framing for optimal performance and reliability.

## Protocol Versions

### Version 1 (V1)
- Basic binary framing
- Message type identification
- Checksum verification
- No compression or encryption

### Version 2 (V2)
- All V1 features
- Optional payload compression (zlib)
- Backward compatible with V1

### Version 3 (V3)
- All V2 features
- Message timestamps
- Priority and broadcast flags
- Acknowledgment requests
- Placeholder for encryption

## Message Structure

### Header Format

```
+--------+--------+--------+----------+----------+-----------+
| Version| Type   | Flags  | Length   | Checksum | Timestamp |
| (1 byte)| (1 byte)| (1 byte)| (4 bytes)| (4 bytes)| (8 bytes) |
+--------+--------+--------+----------+----------+-----------+
```

#### Header Fields

- **Version** (1 byte): Protocol version (1, 2, or 3)
- **Type** (1 byte): Message type identifier
- **Flags** (1 byte): Bit flags for message options
- **Length** (4 bytes): Payload length in bytes (big-endian)
- **Checksum** (4 bytes): CRC32 checksum of payload (big-endian)
- **Timestamp** (8 bytes): Unix timestamp in microseconds (V3 only)

### Header Sizes

- **V1**: 11 bytes (no timestamp)
- **V2**: 11 bytes (no timestamp)
- **V3**: 19 bytes (with timestamp)

## Message Types

### Data Messages

| Type | Hex | Description |
|------|-----|-------------|
| ECG_DATA | 0x01 | ECG signal data |
| ANOMALY_ALERT | 0x02 | Anomaly detection alerts |
| PREDICTION_RESULT | 0x03 | ML prediction results |
| FEATURE_DATA | 0x04 | Extracted features |

### Control Messages

| Type | Hex | Description |
|------|-----|-------------|
| HEARTBEAT | 0x10 | Connection heartbeat |
| PING | 0x11 | Ping request |
| PONG | 0x12 | Ping response |

### Subscription Messages

| Type | Hex | Description |
|------|-----|-------------|
| SUBSCRIBE | 0x20 | Subscribe to data stream |
| UNSUBSCRIBE | 0x21 | Unsubscribe from data stream |
| SUBSCRIPTION_ACK | 0x22 | Subscription acknowledgment |

### System Messages

| Type | Hex | Description |
|------|-----|-------------|
| ERROR | 0xFF | Error message |
| PROTOCOL_NEGOTIATION | 0xFE | Protocol version negotiation |
| VERSION_CHECK | 0xFD | Version compatibility check |

## Flags

### Bit Definitions

| Bit | Flag | Description |
|-----|------|-------------|
| 0 | COMPRESSED | Payload is compressed (V2+) |
| 1 | ENCRYPTED | Payload is encrypted (V3+) |
| 2 | PRIORITY | High priority message (V3+) |
| 3 | BROADCAST | Broadcast to all clients (V3+) |
| 4 | ACK_REQUIRED | Request acknowledgment (V3+) |
| 5-7 | Reserved | Future use |

## Compression

### Algorithm
- **Method**: zlib (DEFLATE)
- **Level**: 6 (balanced compression/speed)
- **Threshold**: Compress payloads > 1KB

### Compression Flow

1. Check if payload size > threshold
2. Apply zlib compression
3. Set COMPRESSED flag
4. Update payload length in header
5. Recalculate checksum

## Checksum

### Algorithm
- **Method**: CRC32
- **Input**: Compressed payload (if compressed) or original payload
- **Verification**: Client must verify checksum before processing

## Timestamps

### Format (V3)
- **Type**: 64-bit unsigned integer
- **Unit**: Microseconds since Unix epoch
- **Precision**: High-resolution timing for performance analysis

## Message Flow

### Connection Establishment

```
Client                    Server
  |                        |
  |--- WebSocket Open ---->|
  |<-- Protocol Negotiation|
  |--- Version Ack ------>|
  |                        |
```

### Data Exchange

```
Client                    Server
  |                        |
  |--- Binary Message --->|
  |<-- Binary Response ---|
  |                        |
```

### Error Handling

```
Client                    Server
  |                        |
  |--- Invalid Message -->|
  |<-- Error Response ----|
  |                        |
```

## MessagePack Integration

### Type Mapping

| Python Type | MessagePack Type | Notes |
|-------------|------------------|-------|
| str | string | UTF-8 encoded |
| int | integer | Variable length |
| float | float | IEEE 754 |
| bool | boolean | True/False |
| None | nil | Null value |
| list | array | Ordered collection |
| dict | map | Key-value pairs |
| datetime | custom | ISO string wrapper |
| uuid.UUID | custom | String wrapper |
| numpy.ndarray | custom | List wrapper |
| bytes | binary | Binary data |

### Custom Type Wrappers

#### DateTime
```json
{
  "__datetime__": "2024-01-15T10:30:45.123456"
}
```

#### UUID
```json
{
  "__uuid__": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### Numpy Array
```json
{
  "__ndarray__": [1.0, 2.0, 3.0],
  "dtype": "float64",
  "shape": [3]
}
```

## Performance Characteristics

### Size Reduction
- **ECG Data**: 50-70% smaller than JSON
- **Metadata**: 40-60% smaller than JSON
- **Binary Data**: Native binary format

### Serialization Speed
- **Encoding**: 2-3x faster than JSON
- **Decoding**: 2-3x faster than JSON
- **Memory**: Lower memory footprint

### Network Efficiency
- **Reduced Bandwidth**: 50-70% less data
- **Lower Latency**: Faster serialization/deserialization
- **Compression**: Additional 30-50% reduction for large payloads

## Error Codes

### Protocol Errors
- **0x100**: Invalid protocol version
- **0x101**: Checksum mismatch
- **0x102**: Message too large
- **0x103**: Decompression failed
- **0x104**: Invalid message type

### Application Errors
- **0x200**: Invalid data format
- **0x201**: Missing required fields
- **0x202**: Authentication failed
- **0x203**: Permission denied
- **0x204**: Rate limit exceeded

## Security Considerations

### Current Implementation
- **Checksum**: Integrity verification
- **Message Validation**: Type checking
- **Size Limits**: Prevent memory exhaustion

### Future Enhancements
- **Encryption**: AES-256-GCM (V3+)
- **Authentication**: Message signing
- **Rate Limiting**: Client throttling

## Compatibility

### Version Support
- **V1 Clients**: Basic functionality
- **V2 Clients**: Compression support
- **V3 Clients**: Full feature set

### Backward Compatibility
- **V3 Server**: Supports V1 and V2 clients
- **V2 Server**: Supports V1 clients
- **V1 Server**: V1 only

### Feature Detection
- **Protocol Negotiation**: Automatic version selection
- **Capability Exchange**: Feature availability
- **Graceful Degradation**: Fallback to supported features

## Implementation Examples

### Python (Server)
```python
from backend.app.websocket.binary_handler import WebSocketBinaryHandler

# Create handler
handler = WebSocketBinaryHandler()

# Send ECG data
await handler.send_message(
    connection_id="client_123",
    message_type=MessageType.ECG_DATA,
    data={
        "signal": [0.1, 0.2, 0.3, ...],
        "timestamp": time.time(),
        "sampling_rate": 250
    }
)
```

### TypeScript (Client)
```typescript
import { WebSocketBinaryService, MessageType } from './websocket-binary.service';

// Create service
const wsService = new WebSocketBinaryService();

// Connect and send data
await wsService.connect('ws://localhost:8000/ws');
wsService.sendMessage(MessageType.ECG_DATA, {
  signal: [0.1, 0.2, 0.3],
  timestamp: Date.now(),
  samplingRate: 250
});
```

## Testing

### Unit Tests
- **Message Encoding/Decoding**: All message types
- **Version Compatibility**: Cross-version testing
- **Compression**: Compression ratio verification
- **Checksum**: Integrity validation
- **Performance**: Speed and size benchmarks

### Integration Tests
- **End-to-End**: Client-server communication
- **Error Handling**: Invalid message processing
- **Concurrent Operations**: Multi-client scenarios
- **Large Messages**: Memory and performance testing

## Monitoring

### Metrics
- **Message Count**: Sent/received per type
- **Message Size**: Average and percentiles
- **Compression Ratio**: Effectiveness metrics
- **Error Rate**: Failed messages
- **Latency**: Processing time distribution

### Logging
- **Connection Events**: Connect/disconnect
- **Protocol Negotiation**: Version selection
- **Errors**: Detailed error information
- **Performance**: Slow message warnings

## Future Enhancements

### Protocol V4 (Planned)
- **Encryption**: AES-256-GCM
- **Message Fragmentation**: Large message support
- **Streaming**: Continuous data streams
- **Quality of Service**: Priority queuing

### Performance Optimizations
- **Zero-Copy**: Direct memory operations
- **Batching**: Multiple messages in one frame
- **Compression**: Alternative algorithms (LZ4)
- **Caching**: Message template reuse

## Reference Implementation

### Server Components
- `messagepack_service.py`: Serialization service
- `binary_protocol.py`: Protocol handler
- `binary_handler.py`: WebSocket integration
- `websocket.py`: Main WebSocket manager

### Client Components
- `websocket-binary.service.ts`: TypeScript client
- `protocol-handler.ts`: Protocol implementation
- `messagepack-utils.ts`: Serialization utilities

## Troubleshooting

### Common Issues
- **Version Mismatch**: Check protocol negotiation
- **Checksum Errors**: Verify data integrity
- **Compression Failures**: Check zlib availability
- **Memory Issues**: Monitor message sizes

### Debug Tools
- **Protocol Analyzer**: Message inspection
- **Performance Monitor**: Latency tracking
- **Size Analyzer**: Compression metrics
- **Error Logger**: Detailed error reporting
