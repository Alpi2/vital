# Backend setup

Quick steps to create a virtual environment and install dependencies for development.

From the project root:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/macOS
# On Windows (PowerShell): .\venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt

# Run the dev server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Notes:

- Use Python 3.10+ for compatibility with pinned packages.
- If you prefer Docker, a containerized recipe can be added to `docker/` to reproduce the build environment.
  Backend (FastAPI)

This folder contains a minimal FastAPI service skeleton.

Run with:

```
pip install fastapi uvicorn
uvicorn app.main:app --reload
```
