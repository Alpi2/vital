# VitalStream Installation Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Requirements](#system-requirements)
3. [Installation Steps](#installation-steps)
4. [Configuration](#configuration)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Software Requirements

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **Kubernetes**: Version 1.24 or higher (for production)
- **Git**: Version 2.30 or higher

### Hardware Requirements

#### Development Environment
- CPU: 4 cores minimum
- RAM: 16 GB minimum
- Storage: 50 GB SSD
- Network: 100 Mbps

#### Production Environment
- CPU: 16 cores minimum (32 cores recommended)
- RAM: 64 GB minimum (128 GB recommended)
- Storage: 1 TB NVMe SSD (RAID 10)
- Network: 1 Gbps dedicated

---

## System Requirements

### Operating System

- **Linux**: Ubuntu 20.04 LTS or higher, RHEL 8+, CentOS 8+
- **macOS**: 11 (Big Sur) or higher
- **Windows**: Windows Server 2019 or higher (with WSL2)

### Database Requirements

- **PostgreSQL**: 15.x with TimescaleDB 2.x extension
- **Redis**: 7.x
- **MongoDB**: 6.x

---

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/vitalstream/vitalstream.git
cd vitalstream
```

### 2. Install Dependencies

#### Rust Dependencies

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
rustup update
```

#### Node.js Dependencies

```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
npm install -g @angular/cli
```

#### Java Dependencies

```bash
sudo apt-get install openjdk-17-jdk
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
```

### 3. Database Setup

#### PostgreSQL with TimescaleDB

```bash
# Add TimescaleDB repository
sudo sh -c "echo 'deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main' > /etc/apt/sources.list.d/timescaledb.list"
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -

# Install
sudo apt-get update
sudo apt-get install timescaledb-2-postgresql-15

# Configure
sudo timescaledb-tune

# Create database
sudo -u postgres psql
CREATE DATABASE vitalstream;
CREATE USER vitalstream_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE vitalstream TO vitalstream_user;
\c vitalstream
CREATE EXTENSION IF NOT EXISTS timescaledb;
\q
```

#### Redis

```bash
sudo apt-get install redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Configure for production
sudo nano /etc/redis/redis.conf
# Set: maxmemory 4gb
# Set: maxmemory-policy allkeys-lru
sudo systemctl restart redis-server
```

#### MongoDB

```bash
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo systemctl enable mongod
sudo systemctl start mongod
```

### 4. Build Backend Services

#### Device Drivers (Rust)

```bash
cd device-drivers
cargo build --release
cd ..
```

#### Alarm Engine (Rust)

```bash
cd alarm-engine
cargo build --release
cd ..
```

#### ML Inference (Rust)

```bash
cd ml-inference
cargo build --release
cd ..
```

#### HL7 Integration (Java)

```bash
cd hl7-integration-service
./mvnw clean package
cd ..
```

### 5. Build Frontend

#### Angular Dashboard

```bash
cd frontend
npm install
npm run build:prod
cd ..
```

#### Qt Central Station

```bash
cd central-station
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ../..
```

### 6. Docker Deployment (Recommended)

```bash
# Build all images
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

### 7. Kubernetes Deployment (Production)

```bash
# Create namespace
kubectl create namespace vitalstream

# Apply configurations
kubectl apply -f k8s/configmaps/
kubectl apply -f k8s/secrets/
kubectl apply -f k8s/deployments/
kubectl apply -f k8s/services/
kubectl apply -f k8s/ingress/

# Verify deployment
kubectl get pods -n vitalstream
```

---

## Configuration

### Environment Variables

Create `.env` file in the root directory:

```bash
# Database
DATABASE_URL=postgresql://vitalstream_user:secure_password@localhost:5432/vitalstream
REDIS_URL=redis://localhost:6379
MONGODB_URL=mongodb://localhost:27017/vitalstream

# Security
JWT_SECRET=your-256-bit-secret-key-here
JWT_EXPIRATION=3600
ENCRYPTION_KEY=your-encryption-key-here

# Services
ALARM_ENGINE_PORT=8081
HL7_SERVICE_PORT=8082
ML_INFERENCE_PORT=8083
API_GATEWAY_PORT=8080

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Logging
LOG_LEVEL=info
LOG_FORMAT=json

# Features
ENABLE_MFA=true
ENABLE_AUDIT_LOG=true
ENABLE_ENCRYPTION=true
```

### Database Migrations

```bash
# Run migrations
cd backend
sqlx migrate run

# Or using Docker
docker-compose exec api sqlx migrate run
```

### SSL/TLS Certificates

```bash
# Generate self-signed certificate (development only)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# For production, use Let's Encrypt
sudo certbot certonly --standalone -d vitalstream.yourhospital.com
```

---

## Verification

### Health Checks

```bash
# API Gateway
curl http://localhost:8080/health

# Alarm Engine
curl http://localhost:8081/health

# HL7 Service
curl http://localhost:8082/health

# ML Inference
curl http://localhost:8083/health
```

### Database Connectivity

```bash
# PostgreSQL
psql -h localhost -U vitalstream_user -d vitalstream -c "SELECT version();"

# Redis
redis-cli ping

# MongoDB
mongosh --eval "db.adminCommand('ping')"
```

### Service Status

```bash
# Docker
docker-compose ps

# Kubernetes
kubectl get pods -n vitalstream
kubectl get services -n vitalstream
```

### Access Web Interface

Open browser and navigate to:
- **Dashboard**: https://localhost:4200
- **API Documentation**: https://localhost:8080/api/docs
- **Grafana**: https://localhost:3000

Default credentials:
- Username: `admin@vitalstream.com`
- Password: `admin123` (change immediately!)

---

## Troubleshooting

### Common Issues

#### Port Already in Use

```bash
# Find process using port
sudo lsof -i :8080

# Kill process
sudo kill -9 <PID>
```

#### Database Connection Failed

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check logs
sudo tail -f /var/log/postgresql/postgresql-15-main.log

# Restart service
sudo systemctl restart postgresql
```

#### Docker Container Won't Start

```bash
# Check logs
docker-compose logs <service_name>

# Rebuild container
docker-compose build --no-cache <service_name>
docker-compose up -d <service_name>
```

#### Kubernetes Pod CrashLoopBackOff

```bash
# Check pod logs
kubectl logs -n vitalstream <pod_name>

# Describe pod
kubectl describe pod -n vitalstream <pod_name>

# Check events
kubectl get events -n vitalstream --sort-by='.lastTimestamp'
```

### Performance Issues

#### High CPU Usage

```bash
# Check top processes
top

# Check Docker stats
docker stats

# Kubernetes resource usage
kubectl top pods -n vitalstream
```

#### High Memory Usage

```bash
# Check memory
free -h

# Check swap
swapon --show

# Clear cache (if needed)
sudo sync && sudo sysctl -w vm.drop_caches=3
```

### Getting Help

- **Documentation**: https://docs.vitalstream.com
- **Support Email**: support@vitalstream.com
- **Community Forum**: https://community.vitalstream.com
- **GitHub Issues**: https://github.com/vitalstream/vitalstream/issues

---

## Next Steps

1. [Configure User Accounts](./USER_MANAGEMENT.md)
2. [Set Up Device Connections](./DEVICE_SETUP.md)
3. [Configure Alarms](./ALARM_CONFIGURATION.md)
4. [Review Security Settings](./SECURITY_GUIDE.md)
5. [Set Up Backups](./BACKUP_GUIDE.md)

---

**Version**: 1.0  
**Last Updated**: January 3, 2026  
**Maintainer**: VitalStream DevOps Team
