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
- **Security:** this service requires a `SECRET_KEY` environment variable for signing tokens. Create a `.env` from `.env.example` and set `SECRET_KEY` before deploying to production. You can generate a secure key locally with:

  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```

  In development you can set `DEBUG=True` in `.env` to enable local dev behavior, but do **not** leave `SECRET_KEY` empty or the app will not start.

- If you prefer Docker, a containerized recipe can be added to `docker/` to reproduce the build environment.

This folder contains a minimal FastAPI service skeleton.

Run with:

```
pip install fastapi uvicorn
uvicorn app.main:app --reload
```

---

**Helper scripts** 🔧

We provide a helper to (re)create the virtualenv, install requirements and run Alembic migrations:

```bash
# Recreate venv, install deps, and run migrations
./backend/scripts/setup_venv.sh --recreate

# Or run without recreation
./backend/scripts/setup_venv.sh
```

This is useful on macOS where system Python can be managed by Homebrew and `pip` may be restricted; the script creates an isolated venv and ensures `alembic` is available before running `alembic upgrade head`.
