# Kong Gateway Setup Guide
# Complete setup and configuration guide

## Overview
Kong API Gateway provides centralized API management for VitalStream microservices.

## Prerequisites
- Docker & Docker Compose
- Kubernetes (for production)
- Redis Cluster
- SSL Certificates

## Quick Start

### Development Setup
```bash
cd infrastructure/kong
docker-compose -f docker-compose.kong-4.0.yml up -d
```

### Production Setup
```bash
kubectl apply -f infrastructure/kubernetes/kong-4.0-deployment.yaml
```

## Configuration Files
- `kong-4.0.yml` - Declarative configuration
- `docker-compose.kong-4.0.yml` - Docker setup
- `kong-4.0-deployment.yaml` - Kubernetes setup

## Services
- **Proxy HTTP**: http://localhost:8000
- **Proxy HTTPS**: https://localhost:8443
- **Admin API**: http://localhost:8001
- **Admin GUI**: http://localhost:8002
- **Metrics**: http://localhost:9542

## Authentication
All API endpoints require JWT authentication. Configure tokens in Kong consumers.

## Rate Limiting
Rate limiting is enforced per service and consumer using Redis cluster.

## Monitoring
- Prometheus metrics: http://localhost:9542/metrics
- Grafana dashboard: http://localhost:3000
- Jaeger tracing: http://localhost:16686

## Troubleshooting
Check logs: `docker-compose logs kong`
Health check: `curl http://localhost:8001/status`
