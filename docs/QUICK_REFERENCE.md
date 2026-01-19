# VitalStream Quick Reference Card

## Common Commands

### Service Management

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Restart a service
docker-compose restart <service_name>

# View logs
docker-compose logs -f <service_name>

# Check status
docker-compose ps
```

### Database Operations

```bash
# Connect to PostgreSQL
psql -U vitalstream_user -d vitalstream

# Backup database
pg_dump -U vitalstream_user vitalstream > backup.sql

# Restore database
psql -U vitalstream_user vitalstream < backup.sql

# Vacuum and analyze
psql -U vitalstream_user -d vitalstream -c "VACUUM ANALYZE;"
```

### Monitoring

```bash
# Check system resources
top
htop
free -h
df -h

# Docker stats
docker stats

# Kubernetes
kubectl top pods -n vitalstream
kubectl get pods -n vitalstream
```

## API Endpoints

### Authentication

```http
POST /api/auth/login
POST /api/auth/logout
POST /api/auth/refresh
```

### Patients

```http
GET    /api/patients
GET    /api/patients/{id}
POST   /api/patients
PUT    /api/patients/{id}
DELETE /api/patients/{id}
```

### Vital Signs

```http
GET /api/patients/{id}/vitals/latest
GET /api/patients/{id}/vitals/history
GET /api/patients/{id}/vitals/trends
```

### Alarms

```http
GET  /api/alarms/active
POST /api/alarms/{id}/acknowledge
GET  /api/alarms/history
```

## Keyboard Shortcuts

### Dashboard

| Shortcut | Action |
|----------|--------|
| `Ctrl+P` | Search patients |
| `Ctrl+A` | View alarms |
| `Ctrl+R` | Generate report |
| `Ctrl+S` | Settings |
| `F5` | Refresh |
| `Esc` | Close modal |

### Waveform Viewer

| Shortcut | Action |
|----------|--------|
| `Space` | Freeze/Unfreeze |
| `+` | Zoom in |
| `-` | Zoom out |
| `←` | Pan left |
| `→` | Pan right |
| `R` | Reset view |

## Alarm Levels

| Level | Color | Priority | Response Time |
|-------|-------|----------|---------------|
| Critical | 🔴 Red | 1 | Immediate |
| High | 🟠 Orange | 2 | <2 minutes |
| Medium | 🟡 Yellow | 3 | <5 minutes |
| Low | 🟢 Green | 4 | <15 minutes |

## Normal Vital Sign Ranges

| Vital Sign | Normal Range | Unit |
|------------|--------------|------|
| Heart Rate | 60-100 | bpm |
| SpO2 | 95-100 | % |
| Systolic BP | 90-120 | mmHg |
| Diastolic BP | 60-80 | mmHg |
| Respiratory Rate | 12-20 | /min |
| Temperature | 36.5-37.5 | °C |

## Configuration Files

```
/opt/vitalstream/
├── .env                    # Environment variables
├── docker-compose.yml      # Docker configuration
├── config/
│   ├── alarm_engine.toml   # Alarm settings
│   ├── database.toml       # Database settings
│   └── security.toml       # Security settings
└── logs/                   # Log files
```

## Log Locations

```
/var/log/vitalstream/
├── api-gateway.log
├── alarm-engine.log
├── device-drivers.log
├── ml-inference.log
└── hl7-integration.log
```

## Port Reference

| Service | Port | Protocol |
|---------|------|----------|
| API Gateway | 8080 | HTTP/HTTPS |
| Alarm Engine | 8081 | HTTP |
| HL7 Service | 8082 | HTTP/MLLP |
| ML Inference | 8083 | HTTP |
| WebSocket | 8080 | WS/WSS |
| PostgreSQL | 5432 | TCP |
| Redis | 6379 | TCP |
| MongoDB | 27017 | TCP |
| Prometheus | 9090 | HTTP |
| Grafana | 3000 | HTTP |

## Emergency Procedures

### System Down

1. Check service status: `docker-compose ps`
2. Check logs: `docker-compose logs`
3. Restart services: `docker-compose restart`
4. If still down, call support: +1 (555) 911-HELP

### Data Loss

1. Stop all services: `docker-compose down`
2. Restore from backup: `./scripts/restore.sh <backup_file>`
3. Verify data: `./scripts/verify_data.sh`
4. Start services: `docker-compose up -d`

### Security Breach

1. Isolate affected systems
2. Change all passwords
3. Review audit logs
4. Contact security team: security@vitalstream.com
5. Follow incident response plan

## Support Contacts

- **Technical Support**: support@vitalstream.com
- **Emergency**: +1 (555) 911-HELP (24/7)
- **Sales**: sales@vitalstream.com
- **Security**: security@vitalstream.com

---

**Version**: 1.0  
**Last Updated**: January 3, 2026
