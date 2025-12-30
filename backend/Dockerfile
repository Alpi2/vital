# Multi-stage Dockerfile for backend (smaller final image)

FROM python:3.11-slim AS builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
ENV PATH=/install/bin:$PATH

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . /app

# Create non-root user
RUN addgroup --system app && adduser --system --ingroup app app || true
USER app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
