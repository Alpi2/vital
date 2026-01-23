# VitalStream Protocol Buffers

This directory contains Protocol Buffer definitions for VitalStream microservices.

## Directory Structure

```
protos/
├── alarm/v1/          # Alarm service definitions
├── device/v1/         # Device management definitions
├── ml/v1/             # ML inference definitions
├── common/v1/          # Common shared definitions
├── buf.yaml           # Buf configuration
└── README.md          # This file
```

## Services

### Alarm Service
- **Package**: `alarm.v1`
- **Files**: 
  - `alarm.proto` - Alarm messages and types
  - `alarm_service.proto` - Alarm service RPC definitions
- **Features**: Real-time alarm management, streaming, batch operations

### Device Service
- **Package**: `device.v1`
- **Files**:
  - `device.proto` - Device messages and types
  - `device_service.proto` - Device service RPC definitions
- **Features**: Device management, data streaming, command execution

### ML Inference Service
- **Package**: `ml.v1`
- **Files**:
  - `inference.proto` - ML inference messages and types
  - `inference_service.proto` - ML service RPC definitions
- **Features**: Model inference, training, deployment, monitoring

### Common Types
- **Package**: `common.v1`
- **Files**:
  - `common.proto` - Common shared messages
  - `error.proto` - Error definitions and handling
  - `pagination.proto` - Pagination and filtering
- **Features**: Shared utilities, error handling, pagination

## Code Generation

### Prerequisites
- Buf CLI installed
- Go runtime (for buf)
- Language-specific tools (protoc, grpc-tools, etc.)

### Generate Code
```bash
# From project root
./scripts/generate_protos.sh

# Or manually
cd protos
buf generate
```

### Generated Code Locations
- **Python**: `../backend/app/generated/protos`
- **Rust**: `../alarm-engine/src/generated`
- **TypeScript**: `../frontend/src/app/generated/protos`
- **Go**: `../go/pkg/protos`
- **Java**: `../hl7-integration-service/src/main/java`
- **C#**: `../dotnet/Generated`

## Versioning

Protocol Buffers use semantic versioning:
- **v1.x.x** - Current stable version
- **Breaking changes** require major version increment
- **Non-breaking changes** increment minor version

## Validation

### Linting
```bash
cd protos
buf lint
```

### Breaking Change Detection
```bash
cd protos
buf breaking --against '.git#branch=main'
```

## Best Practices

1. **Message Design**
   - Use `snake_case` for field names
   - Use `PascalCase` for message names
   - Always include field numbers
   - Reserve field numbers 1-15 for frequently used fields

2. **Service Design**
   - Use descriptive RPC names
   - Include request/response pairs
   - Use streaming for real-time data
   - Include health check RPC

3. **Error Handling**
   - Use standardized error codes from `common.v1.error`
   - Include detailed error messages
   - Provide context and metadata

4. **Versioning**
   - Use package versioning (v1, v2, etc.)
   - Maintain backward compatibility
   - Document breaking changes

## Dependencies

- `google/protobuf/timestamp.proto` - Timestamp handling
- `google/protobuf/duration.proto` - Duration handling
- `google/protobuf/any.proto` - Dynamic message types
- `common/v1/*` - Shared VitalStream types

## Tools and Configuration

### Buf Configuration
- **File**: `buf.yaml`
- **Features**: Linting, breaking change detection, multi-language generation
- **Plugins**: Python, Rust, TypeScript, Go, Java, C#

### Build Script
- **File**: `../scripts/generate_protos.sh`
- **Features**: Automated generation, validation, post-processing
- **Integration**: Git integration, error handling, reporting

## Documentation

Generated documentation is available in:
- **Markdown**: `../docs/protos/`
- **HTML**: Generated via buf documentation plugin

## Support

For questions or issues:
1. Check the [Buf documentation](https://docs.buf.build/)
2. Review the [Protocol Buffers guide](https://developers.google.com/protocol-buffers)
3. Contact the VitalStream development team
