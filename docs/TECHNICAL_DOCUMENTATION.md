# VitalStream Technical Documentation

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Angular    │  │   Qt C++     │  │ Mobile Apps  │     │
│  │  Dashboard   │  │Central Station│  │  iOS/Android │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway Layer                        │
│              (Load Balancer + API Gateway)                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Microservices Layer                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │  Device  │ │  Alarm   │ │    ML    │ │   HL7    │      │
│  │ Drivers  │ │  Engine  │ │Inference │ │Integration│     │
│  │  (Rust)  │ │  (Rust)  │ │  (Rust)  │ │  (Java)  │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  PostgreSQL  │  │    Redis     │  │   MongoDB    │     │
│  │  (Time-series│  │   (Cache)    │  │  (Documents) │     │
│  │     Data)    │  │              │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

#### Backend
- **Rust**: Device drivers, signal processing, ML inference, alarm engine
- **Java**: HL7/FHIR integration, DICOM services
- **Go**: WebRTC telemedicine service
- **Python**: ML model training

#### Frontend
- **Angular 17**: Web dashboard
- **Qt 6 (C++)**: Central monitoring station
- **Swift**: iOS mobile app
- **Kotlin**: Android mobile app

#### Databases
- **PostgreSQL 15**: Primary database with TimescaleDB extension
- **Redis 7**: Caching and real-time data
- **MongoDB 6**: Document storage

#### Infrastructure
- **Kubernetes**: Container orchestration
- **Docker**: Containerization
- **Prometheus + Grafana**: Monitoring
- **ELK Stack**: Logging

---

## Module Documentation

### Device Drivers Module (Rust)

**Location**: `/device-drivers`

**Purpose**: Interface with medical devices via BLE, USB, Serial, and Wi-Fi

**Key Components**:

```rust
// Device trait
pub trait MedicalDevice {
    fn connect(&mut self) -> Result<(), DeviceError>;
    fn disconnect(&mut self) -> Result<(), DeviceError>;
    fn read_data(&mut self) -> Result<VitalSignsData, DeviceError>;
    fn get_device_info(&self) -> DeviceInfo;
}

// Supported devices
- Philips IntelliVue (Serial)
- GE Healthcare CARESCAPE (USB)
- Mindray BeneView (Wi-Fi)
```

**API Endpoints**:

```
POST /api/devices/connect
GET  /api/devices/{device_id}/status
GET  /api/devices/{device_id}/data
POST /api/devices/{device_id}/disconnect
```

### Signal Processing Module (Rust)

**Location**: `/device-drivers/src/signal_processing`

**Features**:
- Baseline wander removal
- Powerline interference filtering (50/60 Hz)
- Motion artifact removal
- Signal quality index (SQI) calculation
- Pan-Tompkins QRS detection

**Example Usage**:

```rust
use vital_device_drivers::signal_processing::*;

let processor = SignalProcessor::new(500); // 500 Hz sampling rate
let filtered = processor.apply_filters(&raw_ecg_data);
let sqi = processor.calculate_sqi(&filtered);
```

### ML Inference Module (Rust)

**Location**: `/ml-inference`

**Models**:
1. **Arrhythmia Detector**: 15+ arrhythmia types
2. **MI Detector**: STEMI/NSTEMI detection
3. **HRV Analyzer**: Time/frequency/nonlinear domain analysis

**Performance**:
- Inference time: <10ms
- Accuracy: >95% on validation set
- Model format: ONNX

**API**:

```rust
let detector = ArrhythmiaDetector::new("models/arrhythmia.onnx")?;
let prediction = detector.predict(&ecg_segment)?;
```

### Alarm Engine (Rust)

**Location**: `/alarm-engine`

**Features**:
- 4-level alarm system (Critical, High, Medium, Low)
- Intelligent alarm filtering
- Escalation logic
- PostgreSQL + Redis persistence

**Configuration**:

```toml
[alarm_engine]
escalation_timeout_secs = 300
max_alarms_per_minute = 10
enable_smart_filtering = true
```

### HL7 Integration (Java)

**Location**: `/hl7-integration-service`

**Supported Messages**:
- ADT^A01: Patient admission
- ADT^A02: Patient transfer
- ADT^A03: Patient discharge
- ORU^R01: Observation results
- ORM^O01: Orders

**Example**:

```java
HL7MessageBuilder builder = new HL7MessageBuilder();
Message oruMessage = builder
    .setMessageType("ORU", "R01")
    .setPatientId("12345")
    .addObservation("HR", "75", "bpm")
    .build();

hl7Sender.send(oruMessage);
```

---

## Database Schema

### PostgreSQL Tables

#### vital_signs

```sql
CREATE TABLE vital_signs (
    id BIGSERIAL PRIMARY KEY,
    patient_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    heart_rate INTEGER,
    spo2 INTEGER,
    systolic_bp INTEGER,
    diastolic_bp INTEGER,
    respiratory_rate INTEGER,
    temperature DECIMAL(4,1),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Hypertable for time-series data
SELECT create_hypertable('vital_signs', 'timestamp');

-- Indexes
CREATE INDEX idx_vitals_patient_time ON vital_signs(patient_id, timestamp DESC);
CREATE INDEX idx_vitals_timestamp_brin ON vital_signs USING BRIN(timestamp);
```

#### alarms

```sql
CREATE TABLE alarms (
    id BIGSERIAL PRIMARY KEY,
    patient_id INTEGER NOT NULL,
    alarm_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by INTEGER,
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_alarms_active ON alarms(patient_id, created_at DESC) 
    WHERE acknowledged = false;
```

### Redis Keys

```
patient:{patient_id}:latest          - Latest vital signs (Hash)
patient:{patient_id}:alarms:active   - Active alarms (List)
session:{session_id}                 - User session (String, TTL: 1h)
cache:vitals:{patient_id}:{timestamp} - Cached vital signs (String, TTL: 5m)
```

---

## API Reference

### Authentication

```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@hospital.com",
  "password": "password123",
  "mfa_code": "123456"
}

Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "expires_in": 3600
}
```

### Vital Signs

```http
GET /api/patients/{patient_id}/vitals/latest
Authorization: Bearer {access_token}

Response:
{
  "patient_id": 123,
  "timestamp": "2026-01-03T10:30:00Z",
  "heart_rate": 75,
  "spo2": 98,
  "blood_pressure": {
    "systolic": 120,
    "diastolic": 80
  },
  "respiratory_rate": 16,
  "temperature": 36.8
}
```

### WebSocket Streaming

```javascript
const ws = new WebSocket('wss://api.vitalstream.com/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    patient_id: 123
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Vital signs:', data);
};
```

---

## Performance Optimization

### Database Optimization

1. **Partitioning**: Time-based partitioning for vital_signs table
2. **Indexing**: BRIN indexes for time-series data
3. **Connection Pooling**: Max 100 connections
4. **Query Optimization**: Use EXPLAIN ANALYZE

### Caching Strategy

- **L1 Cache**: In-memory LRU (1000 entries, 30s TTL)
- **L2 Cache**: Redis (5 min TTL)
- **L3 Cache**: Database

### Network Optimization

- **HTTP/2**: Multiplexing and server push
- **WebSocket**: Binary frames with compression
- **CDN**: Static assets served via CDN

---

## Security

### Authentication & Authorization

- **JWT**: Access tokens (1h expiry)
- **MFA**: TOTP-based two-factor authentication
- **RBAC**: Role-based access control

### Encryption

- **In-Transit**: TLS 1.3
- **At-Rest**: AES-256-GCM
- **Database**: Transparent Data Encryption (TDE)

### Compliance

- **HIPAA**: Full compliance
- **GDPR/KVKK**: Data protection compliance
- **ISO 27001**: Information security management

---

## Deployment

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vitalstream-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vitalstream-api
  template:
    metadata:
      labels:
        app: vitalstream-api
    spec:
      containers:
      - name: api
        image: vitalstream/api:1.0.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

### Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@localhost/vitalstream
REDIS_URL=redis://localhost:6379
JWT_SECRET=your-secret-key
ENVIRONMENT=production
LOG_LEVEL=info
```

---

## Monitoring & Logging

### Prometheus Metrics

```
vitalstream_http_requests_total
vitalstream_http_request_duration_seconds
vitalstream_active_patients
vitalstream_active_alarms
vitalstream_db_connections
```

### Log Format

```json
{
  "timestamp": "2026-01-03T10:30:00Z",
  "level": "INFO",
  "service": "alarm-engine",
  "message": "Alarm triggered",
  "patient_id": 123,
  "alarm_type": "bradycardia"
}
```

---

**Version**: 1.0  
**Last Updated**: January 3, 2026  
**Maintainer**: VitalStream Development Team
