# VitalStream Project Overview

## 📖 Introduction

VitalStream is a real-time ECG (Electrocardiogram) monitoring and analysis system designed to demonstrate modern web technologies in a healthcare context. The project showcases high-performance signal processing using WebAssembly, real-time data streaming, and a responsive medical-grade user interface.

## 🎯 Project Goals

1. **Performance**: Demonstrate WebAssembly's performance advantages for computationally intensive tasks
2. **Real-time Processing**: Implement real-time ECG data streaming and analysis
3. **Modern Architecture**: Showcase a clean three-tier architecture with modern frameworks
4. **Educational**: Serve as a reference implementation for medical software development patterns

## 🏗️ System Architecture

### Three-Tier Architecture

VitalStream follows a classic three-tier architecture:

1. **Presentation Layer** (Frontend)
   - Angular 19 single-page application
   - Real-time ECG waveform visualization using Canvas API
   - Responsive UI with Tailwind CSS
   - WebSocket client for real-time updates

2. **Application Layer** (Backend)
   - FastAPI REST API
   - WebSocket server for real-time data streaming
   - Business logic and data validation
   - Authentication and authorization

3. **Data Layer**
   - SQLite (development) / PostgreSQL (production)
   - Redis for caching and session management
   - File storage for generated reports

### WebAssembly Integration

The core ECG processing algorithms are written in C++ and compiled to WebAssembly using Emscripten:

- **ECG Generation**: Mathematical synthesis of realistic ECG waveforms
- **Signal Processing**: Filtering, noise reduction, and feature extraction
- **Anomaly Detection**: Real-time detection of cardiac anomalies
- **Performance**: 8x faster than equivalent JavaScript implementation

## 🔑 Key Features

### 1. Real-time ECG Monitoring

- Live ECG waveform visualization at 60 FPS
- Multi-patient monitoring support
- Configurable sampling rates and display settings
- Zoom and pan capabilities for detailed analysis

### 2. Anomaly Detection

The system detects various cardiac anomalies:

- **Tachycardia**: Abnormally fast heart rate (>100 BPM)
- **Bradycardia**: Abnormally slow heart rate (<60 BPM)
- **PVC**: Premature Ventricular Contractions
- **Atrial Fibrillation**: Irregular heart rhythm
- **Signal Artifacts**: Noise and interference detection

### 3. Alert System

- Real-time alerts via WebSocket
- Server-Sent Events (SSE) for notifications
- Severity-based alert classification
- Alert history and acknowledgment tracking

### 4. Reporting

- Automated PDF report generation
- Patient statistics and trends
- Anomaly logs and analysis
- Exportable data for further analysis

### 5. Performance Benchmarking

- Built-in benchmark mode
- JavaScript vs C++/WASM performance comparison
- Real-time performance metrics display

## 🛠️ Technology Stack

### Frontend Technologies

- **Angular 19**: Modern web framework with signals and standalone components
- **TypeScript 5**: Type-safe development
- **RxJS 7**: Reactive programming for data streams
- **Chart.js 4**: ECG waveform visualization
- **Tailwind CSS 3**: Utility-first CSS framework
- **WebAssembly**: High-performance signal processing

### Backend Technologies

- **Python 3.11+**: Backend programming language
- **FastAPI**: Modern, fast web framework
- **SQLAlchemy 2**: SQL toolkit and ORM
- **Pydantic 2**: Data validation using Python type annotations
- **Redis**: In-memory data store for caching
- **PostgreSQL 15**: Production database
- **ReportLab**: PDF generation library

### WASM Technologies

- **C++ 17**: WASM source language
- **Emscripten**: C++ to WebAssembly compiler
- **Embind**: C++/JavaScript binding layer

### DevOps & Tools

- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **GitHub Actions**: CI/CD pipeline
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **Nginx**: Reverse proxy and static file serving

## 📊 Data Flow

### ECG Data Processing Pipeline

1. **Data Generation**: Backend generates synthetic ECG data or receives from devices
2. **WebSocket Streaming**: Data streamed to connected clients in real-time
3. **WASM Processing**: Frontend processes data using C++ WASM module
4. **Anomaly Detection**: WASM module detects anomalies and returns results
5. **Visualization**: Processed data rendered on Canvas
6. **Alert Generation**: Anomalies trigger alerts sent to backend
7. **Storage**: Anomalies and session data stored in database

### Authentication Flow

1. User submits credentials to `/api/auth/login`
2. Backend validates credentials against database
3. JWT tokens generated (access + refresh)
4. Tokens stored in Redis for validation
5. Client stores tokens in localStorage
6. Subsequent requests include Bearer token in Authorization header
7. Backend validates token on each request

## 🔒 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: bcrypt with salt
- **CORS Protection**: Configurable allowed origins
- **Rate Limiting**: Protection against abuse
- **Security Headers**: CSP, HSTS, X-Frame-Options, etc.
- **Input Validation**: Pydantic schemas for all inputs
- **SQL Injection Protection**: Parameterized queries via SQLAlchemy

## 📈 Performance Optimizations

### Frontend

- Lazy loading of routes and modules
- Tree shaking and minification
- WebAssembly for compute-intensive tasks
- Canvas rendering for smooth animations
- Virtual scrolling for large datasets

### Backend

- Async I/O with FastAPI
- Database connection pooling
- Redis caching for frequently accessed data
- Efficient database indexing
- Pagination for large result sets

### Database

- Strategic indexes on frequently queried columns
- Composite indexes for complex queries
- Query optimization to prevent N+1 problems

## 🚀 Deployment

### Development Environment

- SQLite database
- Local Redis instance
- Hot reload enabled for both frontend and backend
- Debug logging enabled

### Production Environment

- PostgreSQL database with replication
- Redis cluster for high availability
- Nginx reverse proxy with SSL/TLS
- Docker containers orchestrated with Docker Compose
- Automated backups and monitoring
- Centralized logging

## 📚 Documentation Structure

- **README.md**: Quick start guide and basic information
- **overview.md** (this file): Comprehensive project overview
- **architecture.md**: Detailed architecture documentation
- **USER_GUIDE.md**: End-user documentation
- **setup_dev.md**: Development environment setup
- **API_DOCUMENTATION.md**: API reference
- **CONTRIBUTING.md**: Contribution guidelines

## ⚠️ Important Notes

### Educational Purpose

This project is designed for **demonstration and educational purposes only**. It is not certified for clinical use and should not be used for actual medical diagnosis or treatment.

### Technology Demonstration

The primary goal is to showcase:

- WebAssembly integration in web applications
- Real-time data streaming architectures
- Modern web development best practices
- Performance optimization techniques
- Clean architecture patterns

### Synthetic Data

All ECG data is mathematically generated and does not represent real patient data. The anomaly detection algorithms are simplified for demonstration purposes.

## 🔮 Future Enhancements

Potential areas for expansion:

- Machine learning-based anomaly detection
- Multi-lead ECG support (12-lead)
- Integration with real ECG devices
- Mobile application (React Native/Flutter)
- Advanced signal processing algorithms
- Cloud deployment with auto-scaling
- Multi-tenancy support
- FHIR (Fast Healthcare Interoperability Resources) compliance

## 📞 Support

For questions, issues, or contributions:

- GitHub Issues: Report bugs and request features
- Pull Requests: Contribute code improvements
- Documentation: Help improve documentation

---

**Last Updated**: January 2, 2026  
**Version**: 2.0.0  
**Maintainer**: VitalStream Team
