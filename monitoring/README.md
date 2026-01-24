# VitalStream Monitoring Stack

Bu dizin VitalStream projesi için monitoring ve observability altyapısını içerir.

## Bileşenler

### 1. Prometheus
- **Port:** 9090
- **Amaç:** Metrics toplama ve depolama
- **Retention:** 30 gün
- **Scrape Interval:** 15 saniye

### 2. Grafana
- **Port:** 3000
- **Default Credentials:** admin / admin123
- **Amaç:** Metrics görselleştirme ve dashboard'lar

### 3. Alertmanager
- **Port:** 9093
- **Amaç:** Alert yönetimi ve notification routing

### 4. Exporters

#### Node Exporter (Port: 9100)
- CPU, memory, disk, network metrics
- System-level monitoring

#### PostgreSQL Exporter (Port: 9187)
- Database connection pool
- Query performance
- Table statistics
- Replication lag

#### Redis Exporter (Port: 9121)
- Memory usage
- Hit/miss rates
- Connection statistics

## Kurulum

### 1. Monitoring Stack'i Başlatma

```bash
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

### 2. Servisleri Kontrol Etme

```bash
docker-compose -f docker-compose.monitoring.yml ps
```

### 3. Logları İnceleme

```bash
# Tüm servisler
docker-compose -f docker-compose.monitoring.yml logs -f

# Sadece Prometheus
docker-compose -f docker-compose.monitoring.yml logs -f prometheus

# Sadece Grafana
docker-compose -f docker-compose.monitoring.yml logs -f grafana
```

## Erişim

### Prometheus
- URL: http://localhost:9090
- Targets: http://localhost:9090/targets
- Alerts: http://localhost:9090/alerts

### Grafana
- URL: http://localhost:3000
- Username: `admin`
- Password: `admin123`

### Alertmanager
- URL: http://localhost:9093

## Dashboards

### VitalStream Overview Dashboard

Otomatik olarak provisioned edilen ana dashboard:

**Panels:**
1. **Request Rate** - HTTP request rate by status code
2. **Response Time (p95)** - 95th percentile response time by endpoint
3. **Error Rate** - Percentage of 5xx errors
4. **Active Patients** - Current number of active patients
5. **Anomalies Detected** - Anomalies detected in last hour
6. **System Uptime** - Application uptime
7. **CPU Usage** - Server CPU utilization
8. **Memory Usage** - Server memory utilization
9. **Database Connections** - Active vs max connections
10. **Redis Memory** - Redis memory usage
11. **Anomaly Detection by Type** - Pie chart of anomaly types
12. **API Endpoint Performance** - Table of endpoint metrics

## Metrics

### Backend Metrics

#### HTTP Metrics
```
http_requests_total{method, endpoint, status}
http_request_duration_seconds{method, endpoint}
http_requests_in_progress{method, endpoint}
```

#### Application Metrics
```
active_patients_total
anomalies_detected_total{anomaly_type, severity}
ecg_processing_duration_seconds
wasm_execution_duration_seconds
```

#### Database Metrics
```
db_connections_in_use
db_connections_max
db_query_duration_seconds{query_type}
```

#### Cache Metrics
```
cache_hits_total{cache_type}
cache_misses_total{cache_type}
```

#### Authentication Metrics
```
auth_attempts_total{result}
active_sessions_total
```

### System Metrics (Node Exporter)

```
node_cpu_seconds_total
node_memory_MemTotal_bytes
node_memory_MemAvailable_bytes
node_filesystem_avail_bytes
node_network_receive_bytes_total
node_network_transmit_bytes_total
```

### PostgreSQL Metrics

```
pg_stat_activity_count
pg_stat_activity_max_tx_duration
pg_stat_user_tables_n_dead_tup
pg_stat_user_tables_n_live_tup
pg_replication_lag
```

### Redis Metrics

```
redis_memory_used_bytes
redis_memory_max_bytes
redis_connected_clients
redis_rejected_connections_total
```

## Alerts

### Backend Alerts

1. **HighErrorRate** (Critical)
   - Condition: Error rate > 5%
   - Duration: 5 minutes

2. **HighResponseTime** (Warning)
   - Condition: p95 response time > 1s
   - Duration: 5 minutes

3. **BackendDown** (Critical)
   - Condition: Backend service unreachable
   - Duration: 1 minute

4. **HighCPUUsage** (Warning)
   - Condition: CPU usage > 80%
   - Duration: 10 minutes

5. **HighMemoryUsage** (Warning)
   - Condition: Memory usage > 85%
   - Duration: 10 minutes

6. **DiskSpaceLow** (Warning)
   - Condition: Available disk space < 15%
   - Duration: 5 minutes

### Database Alerts

1. **PostgreSQLDown** (Critical)
   - Condition: PostgreSQL unreachable
   - Duration: 1 minute

2. **PostgreSQLTooManyConnections** (Warning)
   - Condition: Active connections > 80
   - Duration: 5 minutes

3. **PostgreSQLSlowQueries** (Warning)
   - Condition: Query duration > 60s
   - Duration: 5 minutes

4. **RedisDown** (Critical)
   - Condition: Redis unreachable
   - Duration: 1 minute

5. **RedisHighMemoryUsage** (Warning)
   - Condition: Memory usage > 90%
   - Duration: 5 minutes

## Alert Routing

### Channels

1. **#vitalstream-critical** - Critical alerts
2. **#vitalstream-warnings** - Warning alerts
3. **#vitalstream-database** - Database-specific alerts
4. **#vitalstream-backend** - Backend-specific alerts

### Email Notifications

Critical alerts also sent to: `ops-team@example.com`

## Konfigürasyon

### Alertmanager Konfigürasyonu

`alertmanager.yml` dosyasını düzenleyin:

```yaml
global:
  slack_api_url: 'YOUR_SLACK_WEBHOOK_URL'

receivers:
  - name: 'critical-alerts'
    email_configs:
      - to: 'ops-team@example.com'
        auth_username: 'alertmanager@vitalstream.com'
        auth_password: 'YOUR_EMAIL_PASSWORD'
```

### Prometheus Scrape Targets

`prometheus.yml` dosyasında target'ları güncelleyin:

```yaml
scrape_configs:
  - job_name: 'backend'
    static_configs:
      - targets: ['backend:8000']
```

## Bakım

### Data Retention

Prometheus default olarak 30 gün data tutar. Değiştirmek için:

```yaml
command:
  - '--storage.tsdb.retention.time=90d'  # 90 gün
```

### Backup

```bash
# Prometheus data backup
docker run --rm -v monitoring_prometheus-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/prometheus-backup-$(date +%Y%m%d).tar.gz /data

# Grafana data backup
docker run --rm -v monitoring_grafana-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/grafana-backup-$(date +%Y%m%d).tar.gz /data
```

### Restore

```bash
# Prometheus data restore
docker run --rm -v monitoring_prometheus-data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/prometheus-backup-YYYYMMDD.tar.gz -C /

# Grafana data restore
docker run --rm -v monitoring_grafana-data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/grafana-backup-YYYYMMDD.tar.gz -C /
```

## Troubleshooting

### Prometheus Targets Down

1. Container'ların çalıştığını kontrol edin:
   ```bash
   docker ps
   ```

2. Network connectivity kontrol edin:
   ```bash
   docker network inspect monitoring_monitoring
   ```

3. Backend metrics endpoint'ini test edin:
   ```bash
   curl http://localhost:8000/metrics
   ```

### Grafana Dashboard Yüklenmiyor

1. Provisioning dosyalarını kontrol edin:
   ```bash
   docker-compose -f docker-compose.monitoring.yml logs grafana
   ```

2. Dashboard JSON'ını manuel import edin:
   - Grafana UI → Dashboards → Import
   - `grafana/dashboards/vitalstream_overview.json` dosyasını seçin

### Alerts Gönderilmiyor

1. Alertmanager konfigürasyonunu kontrol edin:
   ```bash
   docker-compose -f docker-compose.monitoring.yml exec alertmanager \
     amtool config show
   ```

2. Slack webhook URL'ini test edin:
   ```bash
   curl -X POST -H 'Content-type: application/json' \
     --data '{"text":"Test message"}' \
     YOUR_SLACK_WEBHOOK_URL
   ```

## Best Practices

1. **Alert Fatigue'den Kaçının**
   - Sadece actionable alert'ler oluşturun
   - Threshold'ları gerçekçi belirleyin
   - Inhibition rules kullanın

2. **Dashboard Organization**
   - Her servis için ayrı dashboard
   - Overview dashboard her zaman güncel tutun
   - Drill-down capability sağlayın

3. **Metrics Naming**
   - Prometheus naming conventions'ı takip edin
   - Consistent labeling kullanın
   - Cardinality'ye dikkat edin

4. **Data Retention**
   - Production: 30-90 gün
   - Development: 7-15 gün
   - Long-term storage için external system kullanın

5. **Security**
   - Grafana admin password'ünü değiştirin
   - HTTPS kullanın (production'da)
   - Authentication enable edin
   - Network isolation sağlayın

## Metrics Integration

### Backend'de Metrics Kullanımı

```python
from app.metrics import (
    track_request_metrics,
    record_anomaly,
    update_active_patients
)

@track_request_metrics('/api/v1/patients')
async def get_patients():
    # ... endpoint logic ...
    pass

# Anomaly kaydı
record_anomaly(
    anomaly_type='bradycardia',
    severity='high'
)

# Active patient count güncelleme
update_active_patients(42)
```

## Performance Impact

- Prometheus scraping: ~1-2ms overhead per request
- Metrics storage: ~100MB per day (typical)
- Grafana queries: Minimal impact with proper indexing

## Scaling

### Horizontal Scaling

- Prometheus federation için multiple Prometheus instances
- Grafana için load balancer
- Alertmanager clustering

### Vertical Scaling

- Prometheus için memory artırımı (high cardinality metrics için)
- Grafana için CPU artırımı (complex queries için)

## Monitoring the Monitoring

```promql
# Prometheus health
up{job="prometheus"}

# Scrape duration
prometheus_target_scrape_duration_seconds

# Samples ingested
rate(prometheus_tsdb_head_samples_appended_total[5m])

# Grafana health
up{job="grafana"}
```
