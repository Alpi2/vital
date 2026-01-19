# VitalStream Architecture Documentation

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Design Patterns](#design-patterns)
7. [Security Architecture](#security-architecture)
8. [Performance Optimization](#performance-optimization)
9. [Deployment Architecture](#deployment-architecture)

---

## 🎯 System Overview

VitalStream is a real-time ECG monitoring and anomaly detection system designed for healthcare environments. The system processes ECG data streams, detects anomalies using WebAssembly-accelerated algorithms, and provides real-time alerts to medical professionals.

### Key Features

- **Real-time ECG Monitoring**: Live ECG waveform visualization
- **Anomaly Detection**: ML-powered anomaly detection using WASM
- **Multi-patient Support**: Monitor multiple patients simultaneously
- **Alert System**: Real-time alerts via WebSocket and SSE
- **Historical Analysis**: Patient history and trend analysis
- **Report Generation**: PDF report generation
- **Role-based Access**: Secure authentication and authorization

---

## 🏗️ Architecture Diagram

### High-Level Architecture

```mermaid
flowchart TB
    subgraph Frontend["Frontend Layer"]
        Angular["Angular 19<br/>SPA"]
        WASM["WASM Module<br/>(C++)"]
        WS_Client["WebSocket<br/>Client"]
    end
    
    subgraph Gateway["API Gateway"]
        Nginx["Nginx<br/>Reverse Proxy"]
        RateLimit["Rate Limiter"]
        CORS["CORS Middleware"]
    end
    
    subgraph Backend["Backend Layer"]
        FastAPI["FastAPI<br/>Application"]
        Auth["Auth Service"]
        PatientSvc["Patient Service"]
        AnomalySvc["Anomaly Service"]
        ECGAnalyzer["ECG Analyzer"]
    end
    
    subgraph Data["Data Layer"]
        Postgres[("PostgreSQL<br/>Database")]
        Redis[("Redis<br/>Cache")]
        S3["S3<br/>File Storage"]
    end
    
    subgraph Monitoring["Monitoring"]
        Prometheus["Prometheus"]
        Grafana["Grafana"]
        Sentry["Sentry"]
    end
    
    Angular --> WASM
    Angular --> WS_Client
    Angular --> Nginx
    WS_Client --> Nginx
    
    Nginx --> RateLimit
    RateLimit --> CORS
    CORS --> FastAPI
    
    FastAPI --> Auth
    FastAPI --> PatientSvc
    FastAPI --> AnomalySvc
    FastAPI --> ECGAnalyzer
    
    Auth --> Redis
    PatientSvc --> Postgres
    AnomalySvc --> Postgres
    ECGAnalyzer --> WASM
    
    FastAPI --> S3
    
    FastAPI --> Prometheus
    Prometheus --> Grafana
    FastAPI --> Sentry
```

### Component Interaction

```mermaid
sequenceDiagram
    participant Client
    participant Frontend
    participant API
    participant WASM
    participant DB
    participant WS
    
    Client->>Frontend: Open ECG Monitor
    Frontend->>API: GET /api/patients
    API->>DB: Query patients
    DB-->>API: Patient list
    API-->>Frontend: Patient data
    
    Frontend->>WS: Connect WebSocket
    WS-->>Frontend: Connection established
    
    loop Real-time ECG
        API->>WS: ECG data stream
        WS->>Frontend: Push ECG data
        Frontend->>WASM: Process signal
        WASM-->>Frontend: Processed data
        Frontend->>Client: Update chart
    end
    
    WASM->>Frontend: Anomaly detected
    Frontend->>API: POST /api/anomalies
    API->>DB: Store anomaly
    API->>WS: Broadcast alert
    WS->>Frontend: Alert notification
```

---

## 🔧 Component Details

### Frontend Components

#### 1. **Angular Application**
- **Location**: `frontend/src/app/`
- **Purpose**: Single-page application for ECG monitoring
- **Key Components**:
  - `ECGChartComponent`: Real-time ECG waveform visualization
  - `PatientListComponent`: Patient management interface
  - `AnomalyListComponent`: Anomaly alerts and history
  - `DashboardComponent`: Multi-patient overview

#### 2. **WASM Module**
- **Location**: `wasm/`
- **Purpose**: High-performance ECG signal processing
- **Language**: C++
- **Key Functions**:
  - `process_ecg_data()`: ECG signal filtering and processing
  - `detect_anomalies()`: Real-time anomaly detection
  - `calculate_metrics()`: Heart rate, QRS detection, etc.

#### 3. **Services**
- **ECGDataService**: ECG data management and streaming
- **WasmLoaderService**: WASM module loading and initialization
- **WebSocketService**: Real-time communication
- **AuthService**: Authentication and token management

### Backend Components

#### 1. **API Layer**
- **Location**: `backend/app/api/`
- **Framework**: FastAPI
- **Endpoints**:
  - `/api/auth/*`: Authentication endpoints
  - `/api/patients/*`: Patient management
  - `/api/anomalies/*`: Anomaly detection and alerts
  - `/api/reports/*`: Report generation
  - `/ws/ecg`: WebSocket endpoint for real-time data
  - `/sse/events`: Server-Sent Events for notifications

#### 2. **Service Layer**
- **Location**: `backend/app/services/`
- **Components**:
  - `PatientService`: Patient business logic
  - `AnomalyService`: Anomaly detection and management
  - `ECGAnalyzer`: ECG signal analysis
  - `ReportService`: PDF report generation

#### 3. **Repository Layer**
- **Location**: `backend/app/repositories/`
- **Pattern**: Repository pattern for data access
- **Components**:
  - `PatientRepository`: Patient data access
  - `AnomalyRepository`: Anomaly data access
  - `SessionRepository`: Session management

#### 4. **Middleware**
- **Location**: `backend/app/middleware.py`
- **Components**:
  - `SecurityHeadersMiddleware`: Security headers (CSP, HSTS, etc.)
  - `RequestLoggingMiddleware`: Request/response logging
  - `RateLimitMiddleware`: Rate limiting
  - `ExceptionHandlerMiddleware`: Global error handling

### Database Schema

```mermaid
erDiagram
    PATIENTS ||--o{ ANOMALY_LOGS : has
    PATIENTS ||--o{ ECG_SESSIONS : has
    
    PATIENTS {
        uuid id PK
        varchar name
        int age
        varchar gender
        varchar medical_record_number UK
        timestamp created_at
        timestamp updated_at
    }
    
    ANOMALY_LOGS {
        uuid id PK
        uuid patient_id FK
        timestamp timestamp
        varchar anomaly_type
        varchar severity
        text description
        boolean resolved
        timestamp resolved_at
        timestamp created_at
    }
    
    ECG_SESSIONS {
        uuid id PK
        uuid patient_id FK
        timestamp start_time
        timestamp end_time
        int duration
        varchar status
        timestamp created_at
    }
```

#### Indexes

```sql
-- Anomaly logs indexes
CREATE INDEX idx_anomaly_patient_id ON anomaly_logs(patient_id);
CREATE INDEX idx_anomaly_timestamp ON anomaly_logs(timestamp DESC);
CREATE INDEX idx_anomaly_severity ON anomaly_logs(severity);
CREATE INDEX idx_anomaly_resolved ON anomaly_logs(resolved);
CREATE INDEX idx_anomaly_composite ON anomaly_logs(patient_id, timestamp DESC, severity);

-- Patient indexes
CREATE INDEX idx_patient_mrn ON patients(medical_record_number);
CREATE INDEX idx_patient_created ON patients(created_at DESC);

-- Session indexes
CREATE INDEX idx_session_patient_id ON ecg_sessions(patient_id);
CREATE INDEX idx_session_start_time ON ecg_sessions(start_time DESC);
```

---

## 🔄 Data Flow

### 1. ECG Data Processing Flow

```mermaid
flowchart TD
    A[ECG Device] -->|Raw Data| B[Backend API]
    B --> C[ECG Analyzer Service]
    C --> D[WASM Module]
    D --> E{Anomaly?}
    E -->|Yes| F[Anomaly Detection]
    E -->|No| G[Normal Processing]
    F --> H[Database Storage]
    F --> I[WebSocket Broadcast]
    G --> J[Database Storage]
    I --> K[Connected Clients]
    K --> L[Frontend Update]
```

### 2. Authentication Flow

```mermaid
flowchart TD
    A[User Login] --> B[POST /api/auth/login]
    B --> C{Valid Credentials?}
    C -->|No| D[401 Unauthorized]
    C -->|Yes| E[Generate JWT Tokens]
    E --> F[Access Token 15min]
    E --> G[Refresh Token 7days]
    F --> H[Store in Redis]
    G --> H
    H --> I[Return to Client]
    I --> J[Store in localStorage]
    J --> K[Subsequent Requests]
    K --> L[Auth Interceptor]
    L --> M[Add Bearer Token]
```

### 3. Real-time Alert Flow

```mermaid
flowchart LR
    A[Anomaly Detected] --> B[Database Storage]
    A --> C[WebSocket Broadcast]
    A --> D[SSE Broadcast]
    C --> E[Connected Clients]
    D --> E
    E --> F[Frontend Notification]
    F --> G[Toast Alert]
    F --> H[Sound Alert]
    F --> I[Visual Indicator]
```

---

## 🛠️ Technology Stack

### Frontend

| Technology | Version | Purpose |
|------------|---------|----------|
| Angular | 19.x | Frontend framework |
| TypeScript | 5.x | Type-safe JavaScript |
| RxJS | 7.x | Reactive programming |
| Chart.js | 4.x | ECG visualization |
| WebAssembly | - | High-performance processing |
| Tailwind CSS | 3.x | Styling |
| Playwright | Latest | E2E testing |
| Jasmine/Karma | Latest | Unit testing |

### Backend

| Technology | Version | Purpose |
|------------|---------|----------|
| Python | 3.11+ | Backend language |
| FastAPI | 0.100+ | Web framework |
| SQLAlchemy | 2.x | ORM |
| Pydantic | 2.x | Data validation |
| PyJWT | 2.x | JWT authentication |
| Redis | 7.x | Caching & sessions |
| PostgreSQL | 15+ | Primary database |
| Pytest | Latest | Testing framework |

### WASM

| Technology | Version | Purpose |
|------------|---------|----------|
| C++ | 17+ | WASM source language |
| Emscripten | Latest | WASM build tool |
| Embind | Latest | JS/WASM bindings |

### DevOps

| Technology | Version | Purpose |
|------------|---------|----------|
| Docker | Latest | Containerization |
| Docker Compose | Latest | Multi-container orchestration |
| GitHub Actions | - | CI/CD pipeline |
| Prometheus | Latest | Metrics collection |
| Grafana | Latest | Metrics visualization |
| Nginx | Latest | Reverse proxy |

---

## 🎨 Design Patterns

### 1. Repository Pattern

**Purpose**: Separate data access logic from business logic

**Implementation**:
```python
# backend/app/repositories/patient_repository.py
class PatientRepository:
    """Repository for patient data access operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def get_by_id(self, patient_id: UUID) -> Patient:
        """Retrieve a patient by ID."""
        return await self.db.query(Patient).filter(Patient.id == patient_id).first()
    
    async def create(self, patient: PatientCreate) -> Patient:
        """Create a new patient record."""
        db_patient = Patient(**patient.dict())
        self.db.add(db_patient)
        await self.db.commit()
        return db_patient
```

### 2. Service Layer Pattern

**Purpose**: Encapsulate business logic

**Implementation**:
```python
# backend/app/services/patient_service.py
class PatientService:
    """Service layer for patient business logic."""
    
    def __init__(self, repository: PatientRepository):
        self.repository = repository
    
    async def create_patient(self, patient_data: PatientCreate) -> Patient:
        """Create a new patient with validation."""
        # Business logic here
        return await self.repository.create(patient_data)
```

### 3. Dependency Injection

**Purpose**: Loose coupling and testability

**Implementation**:
```python
# FastAPI dependency injection
@router.post("/patients")
async def create_patient(
    patient: PatientCreate,
    service: PatientService = Depends(get_patient_service)
):
    """Create a new patient endpoint."""
    return await service.create_patient(patient)
```

### 4. Observer Pattern (WebSocket)

**Purpose**: Real-time event broadcasting

**Implementation**:
```python
# backend/app/websocket.py
class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            await connection.send_json(message)
```

### 5. Singleton Pattern (WASM Loader)

**Purpose**: Single instance of WASM module

**Implementation**:
```typescript
// frontend/src/app/services/wasm-loader.service.ts
@Injectable({ providedIn: 'root' })
export class WasmLoaderService {
  private static instance: WasmModule;
  
  async loadWasm(): Promise<WasmModule> {
    if (!WasmLoaderService.instance) {
      WasmLoaderService.instance = await import('../../wasm/pkg');
    }
    return WasmLoaderService.instance;
  }
}
```

---

## 🔒 Security Architecture

### 1. Authentication & Authorization

- **JWT Tokens**: Access token (15 min) + Refresh token (7 days)
- **Token Rotation**: Automatic refresh before expiration
- **Token Blacklist**: Redis-based revocation
- **Password Hashing**: bcrypt with salt

### 2. Security Headers

```python
Content-Security-Policy: default-src 'self'
Strict-Transport-Security: max-age=31536000; includeSubDomains
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
```

### 3. Rate Limiting

- **Global**: 100 requests/minute per IP
- **Auth endpoints**: 5 requests/minute
- **API endpoints**: 60 requests/minute

### 4. Input Validation

- **Pydantic schemas**: Type validation
- **SQL injection**: Parameterized queries (SQLAlchemy)
- **XSS protection**: Content sanitization

---

## ⚡ Performance Optimization

### 1. Frontend Optimizations

- **Lazy Loading**: Route-based code splitting
- **Bundle Size**: Tree shaking, minification
- **Image Optimization**: WebP format, lazy loading
- **WASM**: High-performance signal processing

### 2. Backend Optimizations

- **Database Indexing**: Composite indexes on frequent queries
- **Connection Pooling**: PostgreSQL connection pool (10-20 connections)
- **Caching**: Redis for session data and frequently accessed data
- **Async I/O**: FastAPI async endpoints

### 3. Database Optimizations

- **Indexes**: 9 strategic indexes
- **Query Optimization**: N+1 query prevention
- **Pagination**: Limit/offset for large datasets

---

## 🚀 Deployment Architecture

### Production Environment

```mermaid
flowchart TB
    subgraph LB["Load Balancer"]
        Nginx_LB["Nginx/HAProxy"]
    end
    
    subgraph Frontend_Cluster["Frontend Cluster"]
        FE1["Frontend Container 1"]
        FE2["Frontend Container 2"]
    end
    
    subgraph API_Gateway["API Gateway"]
        Nginx_API["Nginx"]
    end
    
    subgraph Backend_Cluster["Backend Cluster"]
        BE1["Backend Container 1"]
        BE2["Backend Container 2"]
    end
    
    subgraph Data_Layer["Data Layer"]
        PG[("PostgreSQL")]
        RD[("Redis")]
        S3["S3 Storage"]
    end
    
    Nginx_LB --> FE1
    Nginx_LB --> FE2
    FE1 --> Nginx_API
    FE2 --> Nginx_API
    Nginx_API --> BE1
    Nginx_API --> BE2
    BE1 --> PG
    BE1 --> RD
    BE1 --> S3
    BE2 --> PG
    BE2 --> RD
    BE2 --> S3
```

### Container Configuration

**Frontend Container**:
- Base: `nginx:alpine`
- Static files served by Nginx
- Gzip compression enabled
- Cache headers configured

**Backend Container**:
- Base: `python:3.11-slim`
- Gunicorn + Uvicorn workers
- Health check endpoint
- Auto-restart on failure

**Database**:
- PostgreSQL 15+ with replication
- Automated backups (daily)
- Point-in-time recovery

---

## 📊 Monitoring & Observability

### Metrics Collection

- **Prometheus**: Metrics scraping (15s interval)
- **Grafana**: Visualization dashboards
- **Alertmanager**: Alert routing and notifications

### Key Metrics

1. **Application Metrics**:
   - Request rate (req/s)
   - Response time (p50, p95, p99)
   - Error rate (%)
   - Active connections

2. **System Metrics**:
   - CPU usage (%)
   - Memory usage (MB)
   - Disk I/O (MB/s)
   - Network I/O (MB/s)

3. **Database Metrics**:
   - Query duration (ms)
   - Connection pool usage
   - Cache hit rate (%)

### Logging

- **Format**: JSON structured logs
- **Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Aggregation**: Centralized logging (ELK stack compatible)
- **Retention**: 30 days

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

```mermaid
flowchart LR
    A[Push to main] --> B[Test Stage]
    B --> C[Build Stage]
    C --> D[Security Scan]
    D --> E{Branch?}
    E -->|main| F[Deploy Staging]
    E -->|release| G[Deploy Production]
    F --> H[Smoke Tests]
    G --> I[Health Check]
    I --> J{Success?}
    J -->|No| K[Rollback]
    J -->|Yes| L[Complete]
```

**Stages**:
1. **Test**: Backend unit tests, frontend unit tests, E2E tests, code coverage
2. **Build**: Docker image build, WASM compilation, frontend production build
3. **Security Scan**: Dependency vulnerability scan, SAST, container image scan
4. **Deploy**: Staging deployment (auto), production deployment (manual approval), rollback mechanism

---

## 📚 Additional Resources

- [API Documentation](./API_DOCUMENTATION.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
- [Real-time Features](./REALTIME_FEATURES.md)

---

**Last Updated**: January 2, 2026  
**Version**: 2.0.0  
**Maintainer**: VitalStream Team
