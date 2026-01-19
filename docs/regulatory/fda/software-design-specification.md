# Software Design Specification (SDS)

## 1. Introduction

### 1.1 Purpose
This document provides the detailed software design for VitalStream, including architecture, component design, data structures, and interfaces.

### 1.2 Scope
This specification covers:
- System architecture and components
- Detailed component design
- Database schema design
- API specifications
- Security design
- Performance considerations

### 1.3 References
- SRS: Software Requirements Specification
- IEC 62304: Medical Device Software Life Cycle Processes
- ISO 26262: Functional Safety for Automotive (adapted for medical)
- OWASP Top 10: Web Application Security

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VITALSTREAM ARCHITECTURE                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Frontend     │  │   Backend      │  │   Database  │ │
│  │                 │  │                 │  │              │ │
│  │ • Angular UI    │  │ • FastAPI      │  │ • PostgreSQL │ │
│  │ • WebSocket    │  │ • Rust Modules │  │ • Redis     │ │
│  │ • Real-time    │  │ • C++ DICOM   │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   External     │  │   Monitoring   │  │   Storage    │ │
│  │   Integration  │  │                 │  │              │ │
│  │ • HL7 v2.5    │  │ • Prometheus   │  │ • File System│ │
│  │ • DICOM        │  │ • Grafana      │  │ • Cloud      │ │
│  │ • EMR API      │  │ • Alerting    │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Architecture

#### 2.2.1 Frontend Layer
- **Technology:** Angular 17 with TypeScript
- **Responsibilities:** User interface, real-time display, data visualization
- **Components:** Dashboard, ECG Viewer, Alert Manager, Reports

#### 2.2.2 Backend Layer
- **API Gateway:** FastAPI with Python 3.11
- **Processing Engine:** Rust modules for high-performance ECG processing
- **DICOM Handler:** C++ module for medical imaging
- **Business Logic:** Python services for clinical workflows

#### 2.2.3 Data Layer
- **Primary Database:** PostgreSQL 15 for structured data
- **Cache Layer:** Redis 7 for session management and caching
- **File Storage:** Local filesystem with cloud backup

#### 2.2.4 Integration Layer
- **HL7 Interface:** ADT, ORM, and observation messages
- **DICOM Interface:** Image import/export capabilities
- **EMR Integration:** REST API for third-party systems

## 3. Detailed Component Design

### 3.1 Frontend Components

#### 3.1.1 ECG Display Component
```typescript
interface ECGDisplayProps {
  patientId: string;
  leads: ECGLead[];
  samplingRate: number;
  displayDuration: number;
}

interface ECGLead {
  leadId: string;
  data: number[];
  quality: SignalQuality;
  annotations: Annotation[];
}
```

**Design Patterns:**
- Observer Pattern: Real-time data updates
- Strategy Pattern: Different display modes
- Factory Pattern: Lead-specific renderers

#### 3.1.2 Alert Management Component
```typescript
interface Alert {
  id: string;
  patientId: string;
  type: AlertType;
  severity: AlertSeverity;
  timestamp: Date;
  acknowledged: boolean;
  escalated: boolean;
}

enum AlertType {
  ARRHYTHMIA = 'arrhythmia',
  SIGNAL_LOSS = 'signal_loss',
  LEAD_OFF = 'lead_off',
  SYSTEM_ERROR = 'system_error'
}
```

### 3.2 Backend Components

#### 3.2.1 ECG Processing Engine (Rust)
```rust
pub struct ECGProcessor {
    sampling_rate: u32,
    leads: Vec<LeadData>,
    filters: Vec<Box<dyn Filter>>,
    detectors: Vec<Box<dyn Detector>>,
}

impl ECGProcessor {
    pub fn new(sampling_rate: u32) -> Self;
    pub fn add_lead(&mut self, lead: LeadData);
    pub fn process_sample(&mut self, sample: &[f32]) -> Vec<Detection>;
    pub fn get_quality_metrics(&self) -> QualityMetrics;
}

pub trait Detector {
    fn detect(&self, ecg_data: &[f32]) -> Vec<Arrhythmia>;
    fn get_confidence(&self) -> f64;
}
```

#### 3.2.2 API Gateway (FastAPI)
```python
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel

class ECGData(BaseModel):
    patient_id: str
    lead_data: Dict[str, List[float]]
    timestamp: datetime
    quality: SignalQuality

class ArrhythmiaDetection(BaseModel):
    patient_id: str
    arrhythmia_type: str
    confidence: float
    timestamp: datetime
    segment_data: List[float]

app = FastAPI(title="VitalStream API", version="1.0.0")

@app.post("/api/v1/ecg/process")
async def process_ecg(
    ecg_data: ECGData,
    current_user: User = Depends(get_current_user)
) -> ArrhythmiaDetection:
    # Processing logic
    pass
```

### 3.3 Database Design

#### 3.3.1 Core Tables

```sql
-- Patients Table
CREATE TABLE patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mrn VARCHAR(50) UNIQUE NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    date_of_birth DATE NOT NULL,
    gender VARCHAR(10) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ECG Sessions Table
CREATE TABLE ecg_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id),
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ended_at TIMESTAMP WITH TIME ZONE,
    sampling_rate INTEGER NOT NULL,
    leads_config JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ECG Data Table (Time Series)
CREATE TABLE ecg_data (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES ecg_sessions(id),
    lead_id VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    sample_value FLOAT NOT NULL,
    quality_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Arrhythmia Detections Table
CREATE TABLE arrhythmia_detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES ecg_sessions(id),
    detection_time TIMESTAMP WITH TIME ZONE NOT NULL,
    arrhythmia_type VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    algorithm_version VARCHAR(20) NOT NULL,
    reviewed_by UUID REFERENCES users(id),
    confirmed BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Alerts Table
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id),
    session_id UUID REFERENCES ecg_sessions(id),
    detection_id UUID REFERENCES arrhythmia_detections(id),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by UUID REFERENCES users(id),
    escalated_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### 3.3.2 Indexes and Performance

```sql
-- Performance Indexes
CREATE INDEX idx_ecg_data_session_time ON ecg_data(session_id, timestamp);
CREATE INDEX idx_arrhythmia_detections_session_time ON arrhythmia_detections(session_id, detection_time);
CREATE INDEX idx_alerts_patient_severity ON alerts(patient_id, severity, created_at);
CREATE INDEX idx_patients_mrn ON patients(mrn);

-- Partitioning for Time Series Data
CREATE TABLE ecg_data_y2026m01 PARTITION OF ecg_data
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
```

### 3.4 API Specifications

#### 3.4.1 REST API Endpoints

```yaml
openapi: 3.0.0
info:
  title: VitalStream API
  version: 1.0.0
  description: ECG monitoring and analysis system API

paths:
  /api/v1/patients:
    get:
      summary: List patients
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
      responses:
        '200':
          description: List of patients
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Patient'

  /api/v1/ecg/sessions:
    post:
      summary: Create ECG session
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ECGSessionCreate'
      responses:
        '201':
          description: Session created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ECGSession'

  /api/v1/ecg/process:
    post:
      summary: Process ECG data
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ECGData'
      responses:
        '200':
          description: Processing results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ArrhythmiaDetection'

components:
  schemas:
    Patient:
      type: object
      properties:
        id:
          type: string
          format: uuid
        mrn:
          type: string
        firstName:
          type: string
        lastName:
          type: string
        dateOfBirth:
          type: string
          format: date
        gender:
          type: string

    ECGSession:
      type: object
      properties:
        id:
          type: string
          format: uuid
        patientId:
          type: string
          format: uuid
        startedAt:
          type: string
          format: date-time
        samplingRate:
          type: integer
        leadsConfig:
          type: object
```

#### 3.4.2 WebSocket API

```typescript
// WebSocket Message Types
interface WebSocketMessage {
  type: MessageType;
  payload: any;
  timestamp: Date;
}

enum MessageType {
  ECG_DATA = 'ecg_data',
  ALERT = 'alert',
  QUALITY_UPDATE = 'quality_update',
  SESSION_STATUS = 'session_status'
}

// ECG Data Message
interface ECGDataMessage extends WebSocketMessage {
  type: MessageType.ECG_DATA;
  payload: {
    patientId: string;
    sessionId: string;
    leadData: Record<string, number[]>;
    quality: SignalQuality;
  };
}

// Alert Message
interface AlertMessage extends WebSocketMessage {
  type: MessageType.ALERT;
  payload: {
    alertId: string;
    patientId: string;
    alertType: AlertType;
    severity: AlertSeverity;
    message: string;
  };
}
```

## 4. Security Design

### 4.1 Authentication and Authorization

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return await get_user(user_id)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def require_role(required_roles: List[str]):
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role not in required_roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user
    return role_checker
```

### 4.2 Data Encryption

```python
from cryptography.fernet import Fernet
import hashlib
import os

class EncryptionService:
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _get_or_create_key(self) -> bytes:
        key_file = "encryption.key"
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            return key
    
    def encrypt_phi(self, data: str) -> str:
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_phi(self, encrypted_data: str) -> str:
        return self.cipher.decrypt(encrypted_data.encode()).decode()
```

### 4.3 Audit Logging

```python
import logging
from datetime import datetime

class AuditLogger:
    def __init__(self):
        self.logger = logging.getLogger("audit")
        handler = logging.FileHandler("audit.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_access(self, user_id: str, resource: str, action: str):
        self.logger.info(f"USER:{user_id} RESOURCE:{resource} ACTION:{action}")
    
    def log_phi_access(self, user_id: str, patient_id: str, action: str):
        self.logger.warning(f"PHI_ACCESS USER:{user_id} PATIENT:{patient_id} ACTION:{action}")
```

## 5. Performance Considerations

### 5.1 Real-time Processing

```rust
// High-performance ECG processing
use tokio::sync::mpsc;
use std::time::Duration;

pub struct RealTimeProcessor {
    receiver: mpsc::Receiver<ECGData>,
    processors: Vec<Box<dyn Detector>>,
}

impl RealTimeProcessor {
    pub async fn run(&mut self) {
        while let Some(data) = self.receiver.recv().await {
            let start = Instant::now();
            
            // Process with timeout
            let result = tokio::time::timeout(
                Duration::from_millis(100), // 100ms processing limit
                self.process_data(data)
            ).await;
            
            match result {
                Ok(detections) => {
                    let processing_time = start.elapsed();
                    if processing_time > Duration::from_millis(90) {
                        log::warn!("Processing approaching timeout: {:?}", processing_time);
                    }
                    self.send_detections(detections).await;
                }
                Err(_) => {
                    log::error!("Processing timeout for ECG data");
                    self.handle_timeout().await;
                }
            }
        }
    }
}
```

### 5.2 Database Optimization

```sql
-- Materialized Views for Reporting
CREATE MATERIALIZED VIEW daily_patient_summary AS
SELECT 
    p.id,
    p.mrn,
    COUNT(es.id) as session_count,
    MAX(es.started_at) as last_session,
    COUNT(ad.id) as arrhythmia_count
FROM patients p
LEFT JOIN ecg_sessions es ON p.id = es.patient_id
LEFT JOIN arrhythmia_detections ad ON es.id = ad.session_id
WHERE es.started_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY p.id, p.mrn;

-- Refresh Strategy
CREATE OR REPLACE FUNCTION refresh_daily_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW daily_patient_summary;
END;
$$ LANGUAGE plpgsql;

-- Automated Refresh
SELECT cron.schedule('refresh-daily-summary', '0 2 * * *', 'SELECT refresh_daily_summary();');
```

## 6. Error Handling and Recovery

### 6.1 Error Classification

```python
from enum import Enum
from dataclasses import dataclass

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SystemError:
    error_id: str
    severity: ErrorSeverity
    component: str
    message: str
    recovery_action: str
    timestamp: datetime

class ErrorHandler:
    def __init__(self):
        self.error_handlers = {
            ErrorSeverity.CRITICAL: self.handle_critical_error,
            ErrorSeverity.HIGH: self.handle_high_error,
            ErrorSeverity.MEDIUM: self.handle_medium_error,
            ErrorSeverity.LOW: self.handle_low_error,
        }
    
    async def handle_error(self, error: SystemError):
        handler = self.error_handlers.get(error.severity)
        if handler:
            await handler(error)
        
        # Log all errors
        await self.log_error(error)
    
    async def handle_critical_error(self, error: SystemError):
        # Immediate system shutdown
        await self.emergency_shutdown()
        await self.notify_administrators(error)
    
    async def handle_high_error(self, error: SystemError):
        # Component restart
        await self.restart_component(error.component)
        await self.notify_support_team(error)
```

## 7. Testing Strategy

### 7.1 Unit Testing

```python
import pytest
from unittest.mock import Mock, patch

class TestECGProcessor:
    def test_arrhythmia_detection_afib(self):
        # Arrange
        processor = ECGProcessor(sampling_rate=250)
        test_data = self.load_test_ecg("afib_sample.csv")
        
        # Act
        detections = processor.process_segment(test_data)
        
        # Assert
        afib_detections = [d for d in detections if d.type == "AFIB"]
        assert len(afib_detections) > 0
        assert all(d.confidence > 0.9 for d in afib_detections)
    
    def test_signal_quality_assessment(self):
        # Arrange
        processor = ECGProcessor(sampling_rate=250)
        clean_signal = self.generate_clean_signal()
        noisy_signal = self.add_noise(clean_signal)
        
        # Act
        clean_quality = processor.assess_quality(clean_signal)
        noisy_quality = processor.assess_quality(noisy_signal)
        
        # Assert
        assert clean_quality.overall > 0.9
        assert noisy_quality.overall < 0.7
```

### 7.2 Integration Testing

```python
class TestAPIIntegration:
    async def test_ecg_processing_pipeline(self):
        # Arrange
        client = TestClient(app)
        ecg_data = {
            "patient_id": "test-patient-123",
            "lead_data": {"I": [0.1, 0.2, 0.1], "II": [0.2, 0.3, 0.2]},
            "timestamp": "2026-01-18T10:00:00Z"
        }
        
        # Act
        response = client.post("/api/v1/ecg/process", json=ecg_data)
        
        # Assert
        assert response.status_code == 200
        result = response.json()
        assert "arrhythmia_type" in result
        assert "confidence" in result
        assert result["confidence"] > 0.8
```

## 8. Deployment Architecture

### 8.1 Container Configuration

```dockerfile
# Multi-stage Dockerfile
FROM python:3.11-slim as backend
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM node:18-alpine as frontend
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
FROM nginx:alpine
COPY --from=frontend /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
```

### 8.2 Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vitalstream-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vitalstream-backend
  template:
    metadata:
      labels:
        app: vitalstream-backend
    spec:
      containers:
      - name: backend
        image: vitalstream/backend:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: vitalstream-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## 9. Maintenance and Updates

### 9.1 Version Management

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Version:
    major: int
    minor: int
    patch: int
    build: str
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}-{self.build}"

class VersionManager:
    def __init__(self):
        self.current_version = Version(1, 0, 0, "release")
        self.update_history: List[VersionUpdate] = []
    
    def check_compatibility(self, required_version: Version) -> bool:
        if self.current_version.major > required_version.major:
            return True
        if self.current_version.major == required_version.major:
            if self.current_version.minor >= required_version.minor:
                return True
        return False
    
    def plan_update(self, target_version: Version) -> UpdatePlan:
        # Create update plan based on version difference
        steps = []
        if target_version.major > self.current_version.major:
            steps.append(self.create_major_update_plan(target_version))
        elif target_version.minor > self.current_version.minor:
            steps.append(self.create_minor_update_plan(target_version))
        else:
            steps.append(self.create_patch_update_plan(target_version))
        
        return UpdatePlan(
            current_version=self.current_version,
            target_version=target_version,
            steps=steps,
            estimated_duration=self.calculate_update_duration(steps)
        )
```

---

**Document Version:** 1.0  
**Prepared By:** Software Architecture Team  
**Date:** January 4, 2026  
**Next Review:** January 4, 2027
