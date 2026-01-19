# VitalStream User Guide

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [User Interface](#user-interface)
6. [Features](#features)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)
9. [Support](#support)

---

## 🎯 Introduction

VitalStream is a real-time ECG monitoring and anomaly detection system designed for healthcare professionals. This guide will help you get started with installing, configuring, and using VitalStream effectively.

### Who is this guide for?

- **Healthcare Professionals**: Doctors, nurses, and medical staff monitoring patients
- **System Administrators**: IT staff responsible for deployment and maintenance
- **Developers**: Contributors and integrators

### What you'll learn

- How to install and configure VitalStream
- How to monitor patients in real-time
- How to respond to anomaly alerts
- How to generate and export reports
- How to troubleshoot common issues

---

## 🚀 Getting Started

### Prerequisites

Before installing VitalStream, ensure you have:

- **Operating System**: Linux, macOS, or Windows
- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **Web Browser**: Chrome, Firefox, Safari, or Edge (latest version)
- **Network**: Stable internet connection
- **Hardware**: Minimum 4GB RAM, 10GB disk space

### Quick Start (5 minutes)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/vitalstream.git
   cd vitalstream
   ```

2. **Start the application**:
   ```bash
   docker-compose up -d
   ```

3. **Access the application**:
   - Open your browser and navigate to: `http://localhost:4200`
   - Default credentials: `admin` / `admin123`

4. **Verify installation**:
   - You should see the VitalStream dashboard
   - Check that all services are running: `docker-compose ps`

---

## 💻 Installation

### Option 1: Docker Installation (Recommended)

#### Step 1: Install Docker

**Linux (Ubuntu/Debian)**:
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

**macOS**:
```bash
brew install --cask docker
```

**Windows**:
Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop)

#### Step 2: Clone and Configure

```bash
# Clone repository
git clone https://github.com/your-org/vitalstream.git
cd vitalstream

# Copy environment file
cp .env.example .env

# Edit configuration (optional)
nano .env
```

#### Step 3: Start Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

#### Step 4: Initialize Database

```bash
# Run database migrations
docker-compose exec backend alembic upgrade head

# Create admin user
docker-compose exec backend python scripts/create_admin.py
```

### Option 2: Manual Installation

#### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://user:pass@localhost/vitalstream"
export REDIS_URL="redis://localhost:6379"
export SECRET_KEY="your-secret-key-here"

# Run migrations
alembic upgrade head

# Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# Or build for production
npm run build
```

#### WASM Module Setup

```bash
# Navigate to WASM directory
cd wasm

# Install Emscripten (if not installed)
# On macOS:
brew install emscripten

# On Linux:
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh

# Build WASM module
./build_optimized.sh
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Database Configuration
DATABASE_URL=postgresql://vitalstream:password@localhost:5432/vitalstream
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=

# Security
SECRET_KEY=your-super-secret-key-change-this-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS
CORS_ORIGINS=http://localhost:4200,http://localhost:3000

# Monitoring
PROMETHEUS_ENABLED=true
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id

# Email (for alerts)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=noreply@vitalstream.com

# File Storage
S3_BUCKET=vitalstream-reports
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

### Application Settings

Edit `backend/app/config.py` for advanced configuration:

```python
class Settings(BaseSettings):
    # Application
    APP_NAME: str = "VitalStream"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    API_RATE_LIMIT: int = 100  # requests per minute
    
    # ECG Processing
    ECG_SAMPLING_RATE: int = 250  # Hz
    ECG_BUFFER_SIZE: int = 1000  # samples
    ANOMALY_THRESHOLD: float = 0.85  # confidence threshold
    
    # Alerts
    ALERT_COOLDOWN: int = 60  # seconds between same alerts
    CRITICAL_ALERT_SOUND: bool = True
```

---

## 🖥️ User Interface

### Dashboard Overview

The VitalStream dashboard consists of several key sections:

```
┌─────────────────────────────────────────────────────────────┐
│  VitalStream                    [Alerts] [Settings] [Logout] │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Active         │  │  Critical       │  │  Patients    │ │
│  │  Patients: 12   │  │  Alerts: 3      │  │  Monitored   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  Patient List                                         │   │
│  ├───────────────────────────────────────────────────────┤   │
│  │  ID   Name            Status    Heart Rate   Actions  │   │
│  │  001  John Doe        Normal    72 BPM       [View]   │   │
│  │  002  Jane Smith      Alert     95 BPM       [View]   │   │
│  │  003  Bob Johnson     Normal    68 BPM       [View]   │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### ECG Monitoring View

```
┌─────────────────────────────────────────────────────────────┐
│  Patient: John Doe (ID: 001)                    [← Back]     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  ECG Waveform (Real-time)                               │ │
│  │                                                          │ │
│  │      ╱╲    ╱╲    ╱╲    ╱╲    ╱╲                        │ │
│  │  ───╱  ╲──╱  ╲──╱  ╲──╱  ╲──╱  ╲───                    │ │
│  │                                                          │ │
│  │  Heart Rate: 72 BPM  │  Signal Quality: 95%            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Metrics        │  │  Alerts         │  │  History     │ │
│  │  • HR: 72 BPM   │  │  No active      │  │  Last 24h    │ │
│  │  • RR: 850ms    │  │  alerts         │  │  [View]      │ │
│  │  • QT: 380ms    │  │                 │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│                                                               │
│  [Start Recording]  [Generate Report]  [Export Data]         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Navigation

- **Dashboard**: Overview of all patients and system status
- **Patients**: Manage patient records
- **Monitoring**: Real-time ECG monitoring
- **Alerts**: View and manage anomaly alerts
- **Reports**: Generate and download reports
- **Settings**: Configure system preferences
- **Help**: Access documentation and support

---

## 🎨 Features

### 1. Patient Management

#### Adding a New Patient

1. Click **"Patients"** in the navigation menu
2. Click **"Add Patient"** button
3. Fill in patient information:
   - Name (required)
   - Medical Record Number (required, unique)
   - Age
   - Gender
   - Medical history
4. Click **"Save"**

#### Editing Patient Information

1. Navigate to **Patients** list
2. Click **"Edit"** next to the patient
3. Update information
4. Click **"Save Changes"**

#### Deleting a Patient

1. Navigate to **Patients** list
2. Click **"Delete"** next to the patient
3. Confirm deletion
4. **Note**: This will also delete all associated ECG data and alerts

### 2. Real-time ECG Monitoring

#### Starting a Monitoring Session

1. Navigate to **Dashboard**
2. Click **"View"** next to a patient
3. The ECG waveform will start displaying in real-time
4. Monitor the following metrics:
   - **Heart Rate (BPM)**: Beats per minute
   - **RR Interval**: Time between heartbeats
   - **QT Interval**: Ventricular depolarization and repolarization
   - **Signal Quality**: ECG signal quality percentage

#### Understanding the ECG Waveform

```
     R
     │
     │
  P  │  T
  ╱╲ │ ╱╲
 ╱  ╲│╱  ╲
─────┴─────
Q         S

P wave: Atrial depolarization
QRS complex: Ventricular depolarization
T wave: Ventricular repolarization
```

#### Adjusting Display Settings

- **Time Scale**: Adjust horizontal zoom (1s, 5s, 10s, 30s)
- **Amplitude**: Adjust vertical scale (0.5mV, 1mV, 2mV)
- **Speed**: Adjust waveform scroll speed (25mm/s, 50mm/s)
- **Filters**: Enable/disable noise filters

### 3. Anomaly Detection and Alerts

#### Alert Types

VitalStream detects the following anomalies:

| Type | Description | Severity |
|------|-------------|----------|
| **Tachycardia** | Heart rate > 100 BPM | Medium |
| **Bradycardia** | Heart rate < 60 BPM | Medium |
| **Arrhythmia** | Irregular heart rhythm | High |
| **Premature Beat** | Early heartbeat | Low |
| **ST Elevation** | Possible heart attack | Critical |
| **QT Prolongation** | Risk of arrhythmia | High |

#### Responding to Alerts

When an alert is triggered:

1. **Visual Notification**: Red banner appears on screen
2. **Sound Alert**: Audible alarm (for critical alerts)
3. **Toast Notification**: Pop-up message with details

**To respond**:
1. Click on the alert notification
2. Review patient ECG and metrics
3. Take appropriate medical action
4. Mark alert as **"Acknowledged"** or **"Resolved"**

#### Alert Settings

Configure alert preferences in **Settings > Alerts**:

- Enable/disable sound alerts
- Set alert thresholds
- Configure notification recipients
- Set alert cooldown periods

### 4. Report Generation

#### Generating a Patient Report

1. Navigate to patient monitoring view
2. Click **"Generate Report"**
3. Select report type:
   - **Summary Report**: Overview of session
   - **Detailed Report**: Full ECG analysis
   - **Anomaly Report**: List of detected anomalies
4. Select time range (last hour, last 24h, custom)
5. Click **"Generate"**
6. Report will be generated as PDF

#### Report Contents

A typical report includes:

- **Patient Information**: Name, ID, age, gender
- **Session Details**: Start time, duration, status
- **ECG Metrics**: Average HR, RR intervals, QT intervals
- **Anomalies**: List of detected anomalies with timestamps
- **Waveform Snapshots**: Key ECG segments
- **Recommendations**: Automated suggestions

#### Exporting Data

Export raw ECG data for external analysis:

1. Click **"Export Data"**
2. Select format:
   - **CSV**: Comma-separated values
   - **JSON**: JavaScript Object Notation
   - **EDF**: European Data Format (medical standard)
3. Select time range
4. Click **"Download"**

### 5. Multi-Patient Monitoring

#### Dashboard View

Monitor multiple patients simultaneously:

1. Navigate to **Dashboard**
2. View grid of patient cards
3. Each card shows:
   - Patient name and ID
   - Current heart rate
   - Status indicator (Normal/Alert/Critical)
   - Mini ECG waveform
4. Click any card to view detailed monitoring

#### Alert Prioritization

Alerts are prioritized by severity:

1. **Critical** (Red): Immediate attention required
2. **High** (Orange): Urgent attention needed
3. **Medium** (Yellow): Monitor closely
4. **Low** (Blue): Informational

### 6. Historical Data Analysis

#### Viewing Patient History

1. Navigate to patient monitoring view
2. Click **"History"** tab
3. Select time range:
   - Last 24 hours
   - Last 7 days
   - Last 30 days
   - Custom range
4. View:
   - Heart rate trends
   - Anomaly frequency
   - Session summaries

#### Trend Analysis

Analyze long-term patterns:

- **Heart Rate Variability**: Track changes over time
- **Anomaly Patterns**: Identify recurring issues
- **Medication Effects**: Correlate with treatment changes

---

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: Cannot Connect to WebSocket

**Symptoms**:
- ECG waveform not updating
- "Connection lost" message
- No real-time data

**Solutions**:
1. Check network connection
2. Verify backend is running: `docker-compose ps`
3. Check browser console for errors (F12)
4. Ensure WebSocket port (8000) is not blocked by firewall
5. Try refreshing the page (Ctrl+F5)

#### Issue 2: Slow Performance

**Symptoms**:
- Laggy ECG waveform
- Delayed alerts
- Slow page loading

**Solutions**:
1. Check system resources: `docker stats`
2. Increase Docker memory allocation (Settings > Resources)
3. Clear browser cache
4. Reduce number of simultaneous monitoring sessions
5. Check database performance: `docker-compose logs postgres`

#### Issue 3: Authentication Errors

**Symptoms**:
- "Invalid credentials" error
- "Token expired" message
- Automatic logout

**Solutions**:
1. Verify username and password
2. Clear browser cookies and localStorage
3. Check if Redis is running: `docker-compose ps redis`
4. Reset password: `docker-compose exec backend python scripts/reset_password.py`
5. Check token expiration settings in `.env`

#### Issue 4: Missing ECG Data

**Symptoms**:
- Flat line on ECG chart
- No data points
- "No data available" message

**Solutions**:
1. Verify ECG device is connected
2. Check data ingestion logs: `docker-compose logs backend`
3. Verify patient ID is correct
4. Check database connection
5. Restart backend service: `docker-compose restart backend`

#### Issue 5: Alerts Not Triggering

**Symptoms**:
- No alerts despite anomalies
- Missing notifications
- Silent alerts

**Solutions**:
1. Check alert settings in **Settings > Alerts**
2. Verify anomaly detection is enabled
3. Check alert thresholds
4. Review backend logs for errors
5. Test with known anomaly data

### Logs and Debugging

#### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f postgres
docker-compose logs -f redis

# Last 100 lines
docker-compose logs --tail=100 backend
```

#### Debug Mode

Enable debug mode for detailed logging:

```bash
# Edit .env file
DEBUG=true

# Restart services
docker-compose restart
```

#### Health Checks

Check service health:

```bash
# Backend health
curl http://localhost:8000/health

# Database connection
docker-compose exec backend python -c "from app.database import engine; print(engine.connect())"

# Redis connection
docker-compose exec redis redis-cli ping
```

---

## ❓ FAQ

### General Questions

**Q: Is VitalStream HIPAA compliant?**  
A: VitalStream includes security features required for HIPAA compliance (encryption, access controls, audit logs), but full compliance requires proper deployment and configuration. Consult with your compliance team.

**Q: Can I use VitalStream with my existing ECG devices?**  
A: Yes, VitalStream supports standard ECG data formats (EDF, CSV, JSON). You may need to write a custom adapter for proprietary formats.

**Q: How many patients can I monitor simultaneously?**  
A: This depends on your hardware. A typical server can handle 50-100 simultaneous monitoring sessions. For larger deployments, use horizontal scaling.

**Q: Does VitalStream work offline?**  
A: The frontend can cache data for offline viewing, but real-time monitoring requires an active connection to the backend.

### Technical Questions

**Q: What databases are supported?**  
A: PostgreSQL (recommended), MySQL, and SQLite (development only).

**Q: Can I integrate VitalStream with my EHR system?**  
A: Yes, VitalStream provides a REST API and supports HL7 FHIR for integration.

**Q: How is data encrypted?**  
A: Data is encrypted in transit (TLS 1.3) and at rest (AES-256). Database encryption is configurable.

**Q: What's the data retention policy?**  
A: Configurable. Default is 90 days for raw ECG data, 1 year for anomaly logs, indefinite for patient records.

### Troubleshooting Questions

**Q: Why is my ECG waveform choppy?**  
A: This could be due to network latency, insufficient bandwidth, or high CPU usage. Try reducing the sampling rate or increasing buffer size.

**Q: How do I reset the admin password?**  
A: Run: `docker-compose exec backend python scripts/reset_password.py --username admin`

**Q: Can I recover deleted patient data?**  
A: If database backups are enabled, yes. Otherwise, deleted data is permanently removed.

---

## 📞 Support

### Getting Help

- **Documentation**: [https://docs.vitalstream.com](https://docs.vitalstream.com)
- **Community Forum**: [https://community.vitalstream.com](https://community.vitalstream.com)
- **GitHub Issues**: [https://github.com/your-org/vitalstream/issues](https://github.com/your-org/vitalstream/issues)
- **Email Support**: support@vitalstream.com
- **Emergency Hotline**: +1-800-VITAL-STREAM (24/7)

### Reporting Bugs

When reporting bugs, please include:

1. **Description**: What happened vs. what you expected
2. **Steps to Reproduce**: Detailed steps to recreate the issue
3. **Environment**: OS, browser, VitalStream version
4. **Logs**: Relevant log excerpts
5. **Screenshots**: If applicable

### Feature Requests

Submit feature requests via:
- GitHub Issues (label: `enhancement`)
- Community Forum (category: `Feature Requests`)
- Email: features@vitalstream.com

### Contributing

We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## 📄 License

VitalStream is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

---

## 🔄 Updates and Changelog

### Version 2.0.0 (January 2026)

**New Features**:
- Real-time WebSocket streaming
- Enhanced anomaly detection
- Multi-patient dashboard
- PDF report generation
- Mobile-responsive design

**Improvements**:
- 50% faster ECG processing (WASM optimization)
- Better error handling
- Improved security (JWT refresh tokens)
- Enhanced monitoring (Prometheus + Grafana)

**Bug Fixes**:
- Fixed WebSocket reconnection issues
- Resolved memory leaks in long-running sessions
- Fixed timezone handling in reports

See [CHANGELOG.md](../CHANGELOG.md) for full history.

---

**Last Updated**: January 2, 2026  
**Version**: 2.0.0  
**Maintainer**: VitalStream Team

For the latest version of this guide, visit: [https://docs.vitalstream.com/user-guide](https://docs.vitalstream.com/user-guide)
