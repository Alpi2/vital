# VitalStream Troubleshooting Guide

## Quick Reference

| Issue | Quick Fix | Details |
|-------|-----------|----------|
| Service won't start | `docker-compose restart` | [Link](#service-issues) |
| Database connection failed | Check credentials | [Link](#database-issues) |
| High CPU usage | Check logs for loops | [Link](#performance-issues) |
| Alarms not triggering | Verify alarm config | [Link](#alarm-issues) |
| WebSocket disconnects | Check network | [Link](#network-issues) |

---

## Service Issues

### Service Won't Start

**Symptoms**: Docker container exits immediately

**Diagnosis**:
```bash
docker-compose logs <service_name>
docker-compose ps
```

**Solutions**:

1. **Port conflict**:
   ```bash
   sudo lsof -i :<port>
   sudo kill -9 <PID>
   ```

2. **Missing environment variables**:
   ```bash
   # Check .env file
   cat .env
   # Ensure all required variables are set
   ```

3. **Insufficient permissions**:
   ```bash
   sudo chown -R $USER:$USER /opt/vitalstream
   ```

### Service Crashes Repeatedly

**Symptoms**: CrashLoopBackOff in Kubernetes

**Diagnosis**:
```bash
kubectl logs -n vitalstream <pod_name> --previous
kubectl describe pod -n vitalstream <pod_name>
```

**Solutions**:

1. **Memory limit exceeded**:
   ```yaml
   # Increase memory limit in deployment.yaml
   resources:
     limits:
       memory: "2Gi"  # Increase from 1Gi
   ```

2. **Liveness probe failing**:
   ```yaml
   # Adjust probe settings
   livenessProbe:
     initialDelaySeconds: 60  # Increase delay
     periodSeconds: 30
   ```

---

## Database Issues

### Connection Failed

**Symptoms**: "Connection refused" or "Authentication failed"

**Diagnosis**:
```bash
# Test PostgreSQL connection
psql -h localhost -U vitalstream_user -d vitalstream

# Check if PostgreSQL is running
sudo systemctl status postgresql

# Check PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-15-main.log
```

**Solutions**:

1. **PostgreSQL not running**:
   ```bash
   sudo systemctl start postgresql
   sudo systemctl enable postgresql
   ```

2. **Wrong credentials**:
   ```bash
   # Reset password
   sudo -u postgres psql
   ALTER USER vitalstream_user WITH PASSWORD 'new_password';
   ```

3. **Connection limit reached**:
   ```sql
   -- Check current connections
   SELECT count(*) FROM pg_stat_activity;
   
   -- Increase max_connections in postgresql.conf
   -- max_connections = 200
   ```

### Slow Queries

**Symptoms**: API responses taking >1 second

**Diagnosis**:
```sql
-- Find slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Check for missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
AND n_distinct > 100
ORDER BY abs(correlation) DESC;
```

**Solutions**:

1. **Add missing indexes**:
   ```sql
   CREATE INDEX CONCURRENTLY idx_vitals_patient_time 
   ON vital_signs(patient_id, timestamp DESC);
   ```

2. **Vacuum and analyze**:
   ```sql
   VACUUM ANALYZE vital_signs;
   ```

3. **Optimize query**:
   ```sql
   -- Use EXPLAIN ANALYZE to identify bottlenecks
   EXPLAIN ANALYZE SELECT * FROM vital_signs WHERE patient_id = 123;
   ```

---

## Performance Issues

### High CPU Usage

**Symptoms**: CPU usage >80% consistently

**Diagnosis**:
```bash
# Check top processes
top -o %CPU

# Docker stats
docker stats

# Kubernetes
kubectl top pods -n vitalstream
```

**Solutions**:

1. **Infinite loop in code**:
   ```bash
   # Check logs for repeated errors
   docker-compose logs --tail 1000 | grep -i error
   ```

2. **Too many concurrent requests**:
   ```bash
   # Increase replicas
   kubectl scale deployment api-gateway --replicas=5 -n vitalstream
   ```

3. **Inefficient algorithm**:
   ```bash
   # Profile the application
   cargo flamegraph --bin alarm-engine
   ```

### High Memory Usage

**Symptoms**: Memory usage >90%, OOM kills

**Diagnosis**:
```bash
# Check memory
free -h

# Check for memory leaks
valgrind --leak-check=full ./target/release/alarm-engine
```

**Solutions**:

1. **Memory leak**:
   ```rust
   // Use Arc and Weak to prevent circular references
   use std::sync::{Arc, Weak};
   ```

2. **Large cache**:
   ```toml
   # Reduce cache size in config
   [cache]
   l1_size = 500  # Reduce from 1000
   ```

3. **Increase memory limit**:
   ```yaml
   resources:
     limits:
       memory: "4Gi"
   ```

---

## Alarm Issues

### Alarms Not Triggering

**Symptoms**: No alarms despite abnormal vitals

**Diagnosis**:
```bash
# Check alarm engine logs
docker-compose logs alarm-engine --tail 100

# Test alarm endpoint
curl -X POST http://localhost:8081/api/alarms/test \
  -H "Content-Type: application/json" \
  -d '{"patient_id": 123, "heart_rate": 45}'
```

**Solutions**:

1. **Alarm thresholds too wide**:
   ```sql
   -- Check current thresholds
   SELECT * FROM alarm_thresholds WHERE patient_id = 123;
   
   -- Update thresholds
   UPDATE alarm_thresholds 
   SET hr_low = 50, hr_high = 120 
   WHERE patient_id = 123;
   ```

2. **Alarm engine not running**:
   ```bash
   docker-compose restart alarm-engine
   ```

3. **WebSocket connection lost**:
   ```javascript
   // Check WebSocket status in browser console
   console.log(ws.readyState); // Should be 1 (OPEN)
   ```

### Too Many False Alarms

**Symptoms**: Alarm fatigue, frequent false positives

**Solutions**:

1. **Enable smart filtering**:
   ```toml
   [alarm_engine]
   enable_smart_filtering = true
   min_duration_secs = 10  # Require 10s of abnormal values
   ```

2. **Adjust sensitivity**:
   ```rust
   // Increase alarm threshold
   let threshold = AlarmThreshold {
       hr_low: 45,  // Was 50
       hr_high: 125, // Was 120
   };
   ```

---

## Network Issues

### WebSocket Disconnects

**Symptoms**: Frequent "Connection lost" messages

**Diagnosis**:
```bash
# Check network latency
ping api.vitalstream.com

# Check WebSocket logs
docker-compose logs api-gateway | grep -i websocket
```

**Solutions**:

1. **Increase timeout**:
   ```typescript
   const ws = new WebSocket('wss://api.vitalstream.com/ws', {
     handshakeTimeout: 30000  // 30 seconds
   });
   ```

2. **Enable keepalive**:
   ```rust
   // Server-side keepalive
   let config = WebSocketConfig {
       keepalive_interval: Duration::from_secs(30),
   };
   ```

3. **Check firewall**:
   ```bash
   # Allow WebSocket port
   sudo ufw allow 8080/tcp
   ```

### High Latency

**Symptoms**: API responses >500ms

**Diagnosis**:
```bash
# Measure latency
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8080/api/health

# Check network stats
netstat -s | grep -i retrans
```

**Solutions**:

1. **Enable HTTP/2**:
   ```nginx
   listen 443 ssl http2;
   ```

2. **Enable compression**:
   ```nginx
   gzip on;
   gzip_types application/json;
   ```

3. **Use CDN for static assets**:
   ```html
   <script src="https://cdn.vitalstream.com/app.js"></script>
   ```

---

## Device Connection Issues

### Device Not Detected

**Symptoms**: "No device found" error

**Diagnosis**:
```bash
# Check USB devices
lsusb

# Check serial ports
ls -l /dev/ttyUSB*

# Check Bluetooth
hcitool scan
```

**Solutions**:

1. **USB permissions**:
   ```bash
   sudo usermod -a -G dialout $USER
   sudo chmod 666 /dev/ttyUSB0
   ```

2. **Bluetooth pairing**:
   ```bash
   bluetoothctl
   scan on
   pair <MAC_ADDRESS>
   trust <MAC_ADDRESS>
   connect <MAC_ADDRESS>
   ```

3. **Driver not loaded**:
   ```bash
   # Check if driver is running
   docker-compose ps device-drivers
   
   # Restart driver
   docker-compose restart device-drivers
   ```

---

## Frontend Issues

### Blank Screen

**Symptoms**: White screen, no content

**Diagnosis**:
```javascript
// Check browser console for errors
// Press F12 -> Console tab
```

**Solutions**:

1. **JavaScript error**:
   ```bash
   # Rebuild frontend
   cd frontend
   npm run build:prod
   ```

2. **API not accessible**:
   ```javascript
   // Check API endpoint in environment.ts
   export const environment = {
     apiUrl: 'https://api.vitalstream.com'
   };
   ```

3. **Clear browser cache**:
   ```
   Ctrl+Shift+Delete -> Clear cache
   ```

### Waveforms Not Rendering

**Symptoms**: Empty waveform canvas

**Solutions**:

1. **WebGL not supported**:
   ```javascript
   // Check WebGL support
   const canvas = document.createElement('canvas');
   const gl = canvas.getContext('webgl');
   if (!gl) {
     console.error('WebGL not supported');
   }
   ```

2. **Data not streaming**:
   ```javascript
   // Check WebSocket connection
   ws.addEventListener('message', (event) => {
     console.log('Received:', event.data);
   });
   ```

---

## Getting Help

### Before Contacting Support

1. **Collect logs**:
   ```bash
   ./scripts/collect_logs.sh
   ```

2. **Check system info**:
   ```bash
   ./scripts/system_info.sh
   ```

3. **Try safe mode**:
   ```bash
   docker-compose -f docker-compose.safe.yml up
   ```

### Contact Information

- **Email**: support@vitalstream.com
- **Phone**: +1 (555) 123-4567
- **Emergency**: +1 (555) 911-HELP
- **Forum**: https://community.vitalstream.com

---

**Version**: 1.0  
**Last Updated**: January 3, 2026
