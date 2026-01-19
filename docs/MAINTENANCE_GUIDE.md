# VitalStream Maintenance Guide

## Table of Contents

1. [Routine Maintenance](#routine-maintenance)
2. [Database Maintenance](#database-maintenance)
3. [System Updates](#system-updates)
4. [Backup and Recovery](#backup-and-recovery)
5. [Performance Tuning](#performance-tuning)
6. [Security Maintenance](#security-maintenance)

---

## Routine Maintenance

### Daily Tasks

#### 1. System Health Check

```bash
#!/bin/bash
# daily_health_check.sh

echo "=== VitalStream Daily Health Check ==="
echo "Date: $(date)"

# Check service status
echo "\n1. Service Status:"
docker-compose ps

# Check disk space
echo "\n2. Disk Space:"
df -h | grep -E '(Filesystem|/dev/)'

# Check memory
echo "\n3. Memory Usage:"
free -h

# Check database connections
echo "\n4. Database Connections:"
psql -U vitalstream_user -d vitalstream -c "SELECT count(*) FROM pg_stat_activity;"

# Check Redis memory
echo "\n5. Redis Memory:"
redis-cli info memory | grep used_memory_human

# Check active alarms
echo "\n6. Active Alarms:"
curl -s http://localhost:8081/api/alarms/active/count

echo "\n=== Health Check Complete ==="
```

#### 2. Log Review

```bash
# Check for errors in last 24 hours
docker-compose logs --since 24h | grep -i error

# Check alarm engine logs
docker-compose logs alarm-engine --tail 100

# Check API gateway logs
docker-compose logs api-gateway --tail 100
```

#### 3. Backup Verification

```bash
# Verify last backup
ls -lh /backups/vitalstream/ | tail -5

# Check backup integrity
pg_restore --list /backups/vitalstream/latest.dump | head -20
```

### Weekly Tasks

#### 1. Database Vacuum and Analyze

```bash
# Vacuum and analyze all tables
psql -U vitalstream_user -d vitalstream << EOF
VACUUM ANALYZE vital_signs;
VACUUM ANALYZE alarms;
VACUUM ANALYZE patients;
VACUUM ANALYZE users;
EOF
```

#### 2. Index Maintenance

```bash
# Reindex tables
psql -U vitalstream_user -d vitalstream << EOF
REINDEX TABLE vital_signs;
REINDEX TABLE alarms;
EOF
```

#### 3. Log Rotation

```bash
# Rotate application logs
sudo logrotate -f /etc/logrotate.d/vitalstream

# Clean old logs (older than 30 days)
find /var/log/vitalstream -name "*.log" -mtime +30 -delete
```

#### 4. Security Scan

```bash
# Run security scanner
cd /opt/vitalstream
cargo test --package vital-testing security_scanner

# Check for vulnerable dependencies
cargo audit
npm audit
```

### Monthly Tasks

#### 1. Full System Backup

```bash
#!/bin/bash
# monthly_full_backup.sh

BACKUP_DIR="/backups/vitalstream/monthly"
DATE=$(date +%Y%m%d)

mkdir -p $BACKUP_DIR

# Backup PostgreSQL
pg_dump -U vitalstream_user vitalstream | gzip > $BACKUP_DIR/postgres_$DATE.sql.gz

# Backup MongoDB
mongodump --out=$BACKUP_DIR/mongodb_$DATE

# Backup Redis
redis-cli --rdb $BACKUP_DIR/redis_$DATE.rdb

# Backup configuration files
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /opt/vitalstream/config

# Upload to cloud storage
aws s3 sync $BACKUP_DIR s3://vitalstream-backups/monthly/$DATE/

echo "Full backup completed: $DATE"
```

#### 2. Performance Review

```bash
# Generate performance report
psql -U vitalstream_user -d vitalstream << EOF
-- Slowest queries
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;

-- Table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
EOF
```

#### 3. Certificate Renewal

```bash
# Renew Let's Encrypt certificates
sudo certbot renew

# Restart services to apply new certificates
docker-compose restart nginx
```

---

## Database Maintenance

### PostgreSQL Optimization

#### Vacuum Full (Quarterly)

```sql
-- Run during maintenance window
VACUUM FULL vital_signs;
VACUUM FULL alarms;
```

#### Update Statistics

```sql
ANALYZE vital_signs;
ANALYZE alarms;
ANALYZE patients;
```

#### Check for Bloat

```sql
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as index_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Redis Maintenance

#### Memory Optimization

```bash
# Check memory usage
redis-cli info memory

# Clear expired keys
redis-cli --scan --pattern "cache:*" | xargs redis-cli del

# Optimize memory
redis-cli MEMORY PURGE
```

#### Persistence Check

```bash
# Check last save time
redis-cli LASTSAVE

# Force save
redis-cli BGSAVE
```

### MongoDB Maintenance

#### Compact Collections

```javascript
// Connect to MongoDB
mongosh

use vitalstream

// Compact collections
db.runCommand({ compact: 'documents' })
db.runCommand({ compact: 'logs' })
```

#### Rebuild Indexes

```javascript
db.documents.reIndex()
db.logs.reIndex()
```

---

## System Updates

### Update Procedure

#### 1. Pre-Update Checklist

- [ ] Schedule maintenance window
- [ ] Notify users
- [ ] Create full backup
- [ ] Test update in staging environment
- [ ] Prepare rollback plan

#### 2. Update Steps

```bash
# 1. Stop services
docker-compose down

# 2. Backup current version
cp -r /opt/vitalstream /opt/vitalstream.backup.$(date +%Y%m%d)

# 3. Pull latest code
git fetch origin
git checkout v1.1.0  # Replace with target version

# 4. Update dependencies
cd device-drivers && cargo update && cd ..
cd frontend && npm update && cd ..

# 5. Run database migrations
sqlx migrate run

# 6. Rebuild services
docker-compose build

# 7. Start services
docker-compose up -d

# 8. Verify deployment
./scripts/verify_deployment.sh
```

#### 3. Rollback Procedure

```bash
# If update fails, rollback
docker-compose down
rm -rf /opt/vitalstream
mv /opt/vitalstream.backup.$(date +%Y%m%d) /opt/vitalstream
cd /opt/vitalstream
docker-compose up -d
```

---

## Backup and Recovery

### Automated Backup Script

```bash
#!/bin/bash
# /opt/vitalstream/scripts/backup.sh

set -e

BACKUP_DIR="/backups/vitalstream"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

mkdir -p $BACKUP_DIR

echo "Starting backup: $DATE"

# PostgreSQL
echo "Backing up PostgreSQL..."
pg_dump -U vitalstream_user -Fc vitalstream > $BACKUP_DIR/postgres_$DATE.dump

# Redis
echo "Backing up Redis..."
redis-cli --rdb $BACKUP_DIR/redis_$DATE.rdb

# MongoDB
echo "Backing up MongoDB..."
mongodump --out=$BACKUP_DIR/mongodb_$DATE --gzip

# Configuration
echo "Backing up configuration..."
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /opt/vitalstream/config

# Clean old backups
echo "Cleaning old backups..."
find $BACKUP_DIR -name "*.dump" -mtime +$RETENTION_DAYS -delete
find $BACKUP_DIR -name "*.rdb" -mtime +$RETENTION_DAYS -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed: $DATE"
```

### Recovery Procedure

```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: ./restore.sh <backup_file>"
    exit 1
fi

# Stop services
docker-compose down

# Restore PostgreSQL
echo "Restoring PostgreSQL..."
pg_restore -U vitalstream_user -d vitalstream -c $BACKUP_FILE

# Restart services
docker-compose up -d

echo "Restore completed"
```

---

## Performance Tuning

### PostgreSQL Tuning

```sql
-- postgresql.conf optimizations
shared_buffers = 8GB
effective_cache_size = 24GB
maintenance_work_mem = 2GB
work_mem = 64MB
max_connections = 200
random_page_cost = 1.1  -- For SSD
effective_io_concurrency = 200
```

### Redis Tuning

```conf
# redis.conf optimizations
maxmemory 4gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### Application Tuning

```toml
# config/performance.toml
[database]
pool_size = 100
connection_timeout = 30

[cache]
l1_size = 1000
l1_ttl = 30
l2_ttl = 300

[websocket]
max_connections = 10000
buffer_size = 8192
```

---

## Security Maintenance

### Security Audit

```bash
# Run security audit
cargo audit
npm audit

# Check for outdated packages
cargo outdated
npm outdated

# Scan for vulnerabilities
trivy image vitalstream/api:latest
```

### Access Log Review

```bash
# Check failed login attempts
grep "Failed login" /var/log/vitalstream/auth.log | tail -50

# Check suspicious activity
grep -E "(SQL injection|XSS|CSRF)" /var/log/vitalstream/security.log
```

### Certificate Management

```bash
# Check certificate expiry
openssl x509 -in /etc/ssl/certs/vitalstream.crt -noout -dates

# Renew certificates
sudo certbot renew --dry-run
```

---

## Monitoring and Alerts

### Prometheus Alerts

```yaml
# alerts.yml
groups:
  - name: vitalstream
    rules:
      - alert: HighCPUUsage
        expr: cpu_usage > 80
        for: 5m
        annotations:
          summary: "High CPU usage detected"
      
      - alert: HighMemoryUsage
        expr: memory_usage > 90
        for: 5m
        annotations:
          summary: "High memory usage detected"
      
      - alert: DatabaseConnectionPoolExhausted
        expr: db_connections > 180
        for: 1m
        annotations:
          summary: "Database connection pool nearly exhausted"
```

---

**Version**: 1.0  
**Last Updated**: January 3, 2026  
**Maintainer**: VitalStream Operations Team
