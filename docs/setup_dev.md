# VitalStream Development Environment Setup

This guide will help you set up a complete development environment for VitalStream.

## 📋 Prerequisites

### Required Software

- **Node.js** 18+ and npm
- **Python** 3.11+
- **Docker** and Docker Compose
- **Git**
- **Code Editor** (VS Code recommended)

### Optional Tools

- **Emscripten** (for WASM development)
- **PostgreSQL** client tools
- **Redis** client tools
- **Postman** or similar API testing tool

## 🚀 Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/vital.git
cd vital
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env

# Edit .env and set your configuration
# Important: Set a secure SECRET_KEY
nano .env  # or use your preferred editor

# Initialize database
python -m app.init_db

# Run migrations (if using Alembic)
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

The frontend will be available at `http://localhost:4200`

### 4. WASM Module Setup (Optional)

If you need to rebuild the WASM module:

#### Using Docker (Recommended)

```bash
cd core
docker build -f docker/Dockerfile.emsdk -t emsdk-builder .
docker run --rm -v $(pwd)/../frontend/src/assets/wasm:/output emsdk-builder bash /src/scripts/build-wasm-local.sh
```

#### Using Local Emscripten

```bash
# Install Emscripten
# On macOS:
brew install emscripten

# On Linux:
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh

# Build WASM module
cd scripts
./build-wasm-local.sh
```

## 🐳 Docker Development Setup

### Full Stack with Docker Compose

```bash
# Create environment file
cp backend/.env.example backend/.env
# Edit backend/.env and set SECRET_KEY

# Build and start all services
docker compose up --build

# Or run in detached mode
docker compose up -d --build

# View logs
docker compose logs -f

# Stop services
docker compose down
```

Services will be available at:
- Frontend: `http://localhost:4200`
- Backend API: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`

## 🗄️ Database Setup

### SQLite (Default for Development)

SQLite is used by default and requires no additional setup. The database file will be created automatically at `backend/data/vitalstream.db`.

### PostgreSQL (Optional for Development)

```bash
# Start PostgreSQL with Docker
docker compose up postgres -d

# Update backend/.env
DATABASE_URL=postgresql+asyncpg://vitalstream:securepassword@localhost:5432/vitalstream

# Run migrations
cd backend
alembic upgrade head

# Create initial data (optional)
python -m app.seed_data
```

### Redis Setup

```bash
# Start Redis with Docker
docker compose up redis -d

# Update backend/.env
REDIS_URL=redis://localhost:6379/0

# Test Redis connection
redis-cli ping
```

## 🔧 IDE Configuration

### VS Code Recommended Extensions

```json
{
  "recommendations": [
    "angular.ng-template",
    "ms-python.python",
    "ms-python.vscode-pylance",
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "ms-azuretools.vscode-docker",
    "redhat.vscode-yaml",
    "ms-vscode.cpptools"
  ]
}
```

### VS Code Settings

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/backend/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[html]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

## 🧪 Running Tests

### Backend Tests

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_patients.py

# Run with verbose output
pytest -v
```

### Frontend Tests

```bash
cd frontend

# Run unit tests
npm test

# Run tests with coverage
npm run test:coverage

# Run E2E tests
npm run e2e

# Run E2E tests in headless mode
npm run e2e:headless
```

## 🔍 Debugging

### Backend Debugging (VS Code)

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "app.main:app",
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        "8000"
      ],
      "jinja": true,
      "justMyCode": false,
      "cwd": "${workspaceFolder}/backend",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/backend"
      }
    }
  ]
}
```

### Frontend Debugging (Chrome DevTools)

1. Start the development server: `npm start`
2. Open Chrome DevTools (F12)
3. Go to Sources tab
4. Set breakpoints in TypeScript files
5. Refresh the page to trigger breakpoints

### WASM Debugging

```bash
# Build WASM with debug symbols
emcc -g4 -s ASSERTIONS=2 -s SAFE_HEAP=1 ...

# Use Chrome DevTools to debug WASM
# Enable DWARF support in Chrome flags:
# chrome://flags/#enable-webassembly-debugging
```

## 📊 Development Tools

### API Documentation

FastAPI provides automatic API documentation:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

### Database Management

```bash
# SQLite
sqlite3 backend/data/vitalstream.db

# PostgreSQL
psql -h localhost -U vitalstream -d vitalstream

# Using GUI tools
# - DBeaver (cross-platform)
# - pgAdmin (PostgreSQL)
# - DB Browser for SQLite
```

### Redis Management

```bash
# Redis CLI
redis-cli

# Monitor Redis commands
redis-cli monitor

# Using GUI tools
# - RedisInsight
# - Redis Commander
```

## 🔄 Hot Reload

### Backend Hot Reload

Uvicorn automatically reloads when Python files change (when using `--reload` flag).

### Frontend Hot Reload

Angular CLI automatically reloads when TypeScript/HTML/CSS files change.

### WASM Hot Reload

WASM files need to be manually rebuilt. After rebuilding, refresh the browser.

## 🌍 Environment Variables

### Backend Environment Variables

Create `backend/.env`:

```bash
# Application
SECRET_KEY=your-secret-key-here-change-in-production
DEBUG=True
ENVIRONMENT=development

# Database
DATABASE_URL=sqlite+aiosqlite:///./data/vitalstream.db
# Or for PostgreSQL:
# DATABASE_URL=postgresql+asyncpg://vitalstream:securepassword@localhost:5432/vitalstream

# Redis
REDIS_URL=redis://localhost:6379/0

# CORS
CORS_ORIGINS=http://localhost:4200,http://localhost:4201

# JWT
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# Logging
LOG_LEVEL=DEBUG
```

### Frontend Environment Variables

Angular uses `environment.ts` files (not `.env`):

```typescript
// frontend/src/environments/environment.ts
export const environment = {
  production: false,
  apiUrl: 'http://localhost:8000/api',
  wsUrl: 'ws://localhost:8000/ws',
};
```

## 🛠️ Common Development Tasks

### Adding a New API Endpoint

1. Create route in `backend/app/api/`
2. Add service logic in `backend/app/services/`
3. Add repository methods in `backend/app/repositories/`
4. Add Pydantic schemas in `backend/app/schemas/`
5. Write tests in `backend/tests/`

### Adding a New Frontend Component

```bash
cd frontend
ng generate component components/my-component
```

### Creating Database Migrations

```bash
cd backend

# Auto-generate migration from model changes
alembic revision --autogenerate -m "Add new table"

# Apply migration
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Updating Dependencies

```bash
# Backend
cd backend
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt

# Frontend
cd frontend
npm update
npm audit fix
```

## 🐛 Troubleshooting

### Backend Issues

**Issue**: `ModuleNotFoundError: No module named 'app'`

```bash
# Solution: Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"
```

**Issue**: Database connection errors

```bash
# Check DATABASE_URL in .env
# Ensure database service is running
docker compose up postgres -d
```

### Frontend Issues

**Issue**: `Cannot find module '@angular/core'`

```bash
# Solution: Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

**Issue**: CORS errors

```bash
# Solution: Check CORS_ORIGINS in backend/.env
# Ensure it includes your frontend URL
CORS_ORIGINS=http://localhost:4200
```

### WASM Issues

**Issue**: WASM module fails to load

```bash
# Solution: Rebuild WASM module
cd scripts
./build-wasm-local.sh

# Clear browser cache and reload
```

### Docker Issues

**Issue**: Port already in use

```bash
# Solution: Stop conflicting services
docker compose down
# Or change ports in docker-compose.yml
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Angular Documentation](https://angular.io/docs)
- [Emscripten Documentation](https://emscripten.org/docs/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Docker Documentation](https://docs.docker.com/)

## 🤝 Getting Help

- Check existing GitHub issues
- Read the documentation in `docs/`
- Ask questions in GitHub Discussions
- Review the code examples in the repository

---

**Last Updated**: January 2, 2026  
**Version**: 2.0.0  
**Maintainer**: VitalStream Team
