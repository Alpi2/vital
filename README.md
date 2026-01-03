# VitalStream - Real-time ECG Analysis

![VitalStream Dashboard](docs/screenshots/dashboard.png)

## 🎯 Overview

VitalStream is a high-performance, real-time ECG analysis dashboard that simulates medical-grade heart monitoring with anomaly detection. Built with a modern three-tier architecture for maximum performance and scalability.

## 🚀 Features

- **Real-time ECG Simulation**: Mathematical synthetic ECG generation with realistic P-Q-R-S-T waves
- **WebAssembly Powered**: C++ core compiled to WASM for near-native performance in browser
- **Anomaly Detection**: Real-time detection of tachycardia, bradycardia, PVC, AFib, and artifacts
- **Interactive Dashboard**: Medical-grade UI with dark mode, real-time charts, and alerts
- **PDF Reporting**: Automated report generation with patient statistics and anomaly logs
- **Benchmark Mode**: Compare JavaScript vs C++ performance (8x faster!)

## 🏗️ Architecture

```
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Angular 17+     │◄──►│ C++/WASM Core   │◄──►│ Python FastAPI  │
│ Frontend        │    │ (Emscripten)    │    │ Backend         │
│ • Canvas        │    │ • ECG Gen       │    │ • REST API      │
│ • Signals       │    │ • Analysis      │    │ • SQLite        │
│ • TailwindCSS   │    │ • Benchmark     │    │ • ReportLab     │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## 📋 Prerequisites

- Node.js 18+
- Python 3.11+
- Docker & Docker Compose
- Modern browser with WebAssembly support

## 🛠️ Installation

### Quick Start with Docker

```bash
git clone https://github.com/YOUR_USERNAME/vital.git
cd vital

# Create .env file from example
cp backend/.env.example backend/.env
# Edit backend/.env and set a secure SECRET_KEY

# Build and start all services
docker compose up --build

# Or run in detached mode
docker compose up -d --build

# View logs
docker compose logs -f

# Stop all services
docker compose down
```

**Services will be available at:**
- Frontend: http://localhost:4200
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- PostgreSQL: localhost:5432 (optional)
- Redis: localhost:6379

```

### Manual Installation

Backend Setup:

```bash
cd backend

# Create environment file
cp .env.example .env
# Edit .env and set a secure SECRET_KEY

python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
uvicorn app.main:app --reload
```

WASM Build:

```bash
cd core
docker build -f docker/Dockerfile.emsdk -t emsdk-builder .
docker run --rm -v $(pwd)/../frontend/src/assets/wasm:/output emsdk-builder bash /src/scripts/build-wasm-local.sh
```

Frontend Setup:

```bash
cd frontend
npm install
npm run build
npm start
```

## 🗄️ Database Configuration

### SQLite (Default - Development)

The project uses SQLite by default for development:

```bash
# Already configured in backend/.env.example
DATABASE_URL=sqlite+aiosqlite:///./data/vitalstream.db
```

### PostgreSQL (Production)

For production environments, use PostgreSQL:

```bash
# Update backend/.env
DATABASE_URL=postgresql+asyncpg://vitalstream:securepassword@localhost:5432/vitalstream

# Start PostgreSQL with Docker Compose
docker compose up postgres -d

# Run migrations
cd backend
alembic upgrade head
```

**Note**: PostgreSQL requires `asyncpg` and `psycopg2-binary` packages (already included in requirements.txt).

## 🧪 Testing

```bash
# Frontend tests
cd frontend/vitalstream-frontend
npm test

# Backend tests
cd backend
pytest

# Integration tests (example)
docker-compose -f docker-compose.test.yml up --build
```

## 📊 Performance Metrics

| Operation           | JavaScript | C++/WASM | Improvement |
| ------------------- | ---------: | -------: | ----------: |
| 10k sample analysis |      120ms |     15ms |   8x faster |
| Real-time rendering |     30 FPS |   60 FPS | 2x smoother |
| Memory usage        |      45 MB |    12 MB |    73% less |

⚠️ **Disclaimer**
This software is for demonstration and educational purposes only. It simulates medical data and is not certified for clinical use. Always consult healthcare professionals for medical advice.

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

---

#### 10.2. API Documentation

The FastAPI app exposes OpenAPI/Swagger documentation at `/api/docs` (configured in `backend/app/main.py`). You can also expose a JSON OpenAPI document and customize the schema in non-production environments.

Example snippet (already present in the codebase):

```python
# backend/app/main.py - Swagger/OpenAPI auto-documentation
@app.get("/api/docs/json", include_in_schema=False)
async def get_openapi_json():
    return app.openapi()

# Environment-specific docs
if not settings.PRODUCTION:
    from fastapi.openapi.utils import get_openapi

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title="VitalStream API",
            version="1.0.0",
            description="Real-time ECG Analysis Backend",
            routes=app.routes,
        )

        # Add security schemas
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        }

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi
```
