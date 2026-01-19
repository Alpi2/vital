# Runbook: Database Connection Pool Exhausted

## Severity
**CRITICAL**

## Description
Database connection pool is exhausted, preventing applications from establishing new database connections. This can cause service degradation or complete service failure.

## Impact
- **Service Availability**: Applications cannot connect to database
- **Data Operations**: Read/write operations fail
- **User Experience**: Service errors and timeouts
- **Data Integrity**: Potential data loss if writes fail

## Diagnosis

### 1. Check Grafana Dashboard
- **Dashboard**: [System Health Dashboard](https://grafana.vitalstream.com/d/vitalstream-system-health)
- **Key Metrics to Check**:
  - Database Connections Active
  - Connection Pool Usage
  - Database Query Duration
  - Database Error Rate

### 2. Check Prometheus Metrics
```bash
# Check active connections
curl -s "http://prometheus:9090/api/v1/query?query=database_connections_active{environment=\"production\"}"

# Check connection pool usage
curl -s "http://prometheus:9090/api/v1/query?query=database_connections_active{environment=\"production\"} / database_connections_max{environment=\"production\"} * 100"

# Check connection wait time
curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(database_connection_wait_time_seconds_bucket{environment=\"production\"}[5m]))"
```

### 3. Check Database Health
```bash
# Check PostgreSQL connection
kubectl exec -it deployment/postgres -n vitalstream -- psql -U postgres -d vitalstream -c "SELECT count(*) FROM pg_stat_activity;"

# Check connection limits
kubectl exec -it deployment/postgres -n vitalstream -- psql -U postgres -d vitalstream -c "SHOW max_connections;"

# Check slow queries
kubectl exec -it deployment/postgres -n vitalstream -- psql -U postgres -d vitalstream -c "
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
WHERE mean_time > 1000
ORDER BY mean_time DESC
LIMIT 10;
"
```

### 4. Check Application Connection Pool
```bash
# Check application logs for connection errors
kubectl logs -l app=vitalstream-backend -n vitalstream | grep -E "(connection|pool|timeout)" | tail -20

# Check connection pool configuration
kubectl get configmap backend-config -n vitalstream -o yaml | grep -A 10 database

# Test database connection from application
kubectl exec -it deployment/vitalstream-backend -n vitalstream -- python -c "
import psycopg2
import time
start = time.time()
try:
    conn = psycopg2.connect('postgresql://postgres:password@postgres:5432/vitalstream')
    print(f'Connection successful in {time.time()-start:.2f}s')
    conn.close()
except Exception as e:
    print(f'Connection failed: {e}')
"
```

### 5. Check Database Performance
```bash
# Check database locks
kubectl exec -it deployment/postgres -n vitalstream -- psql -U postgres -d vitalstream -c "
SELECT blocked_locks.pid AS blocked_pid,
       blocked_activity.usename AS blocked_user,
       blocking_locks.pid AS blocking_pid,
       blocking_activity.usename AS blocking_user,
       blocked_activity.query AS blocked_statement,
       blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
"

# Check database size and table sizes
kubectl exec -it deployment/postgres -n vitalstream -- psql -U postgres -d vitalstream -c "
SELECT schemaname,
       tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;
"
```

## Resolution Steps

### Step 1: Immediate Triage (First 5 minutes)
1. **Verify Alert Accuracy**
   ```bash
   # Check current connection pool usage
   curl -s "http://prometheus:9090/api/v1/query?query=database_connections_active{environment=\"production\"} / database_connections_max{environment=\"production\"} * 100"
   ```

2. **Identify Connection Consumers**
   ```bash
   # Check which applications are using connections
   kubectl exec -it deployment/postgres -n vitalstream -- psql -U postgres -d vitalstream -c "
   SELECT datname, usename, count(*) AS connection_count
   FROM pg_stat_activity
   GROUP BY datname, usename
   ORDER BY connection_count DESC;
   "
   ```

3. **Check for Long-Running Queries**
   ```bash
   # Check long-running queries
   kubectl exec -it deployment/postgres -n vitalstream -- psql -U postgres -d vitalstream -c "
   SELECT pid, now() - pg_stat_activity.query_start AS duration, query
   FROM pg_stat_activity
   WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
   ORDER BY duration DESC;
   "
   ```

### Step 2: Connection Pool Optimization (5-15 minutes)
1. **Increase Connection Pool Size**
   ```bash
   # Update application connection pool configuration
   kubectl patch configmap backend-config -n vitalstream -p '{
     "data": {
       "database.yaml": "max_connections: 50\\nmin_connections: 5\\nmax_overflow: 20\\nconnection_timeout: 30\\nrecycle: 3600"
     }
   }'
   
   # Restart application to apply changes
   kubectl rollout restart deployment/vitalstream-backend -n vitalstream
   ```

2. **Increase Database Max Connections**
   ```bash
   # Update PostgreSQL configuration
   kubectl patch configmap postgres-config -n vitalstream -p '{
     "data": {
       "postgresql.conf": "max_connections = 200\\nshared_buffers = 256MB\\neffective_cache_size = 1GB\\nwork_mem = 4MB\\nmaintenance_work_mem = 64MB"
     }
   }'
   
   # Restart PostgreSQL to apply changes
   kubectl rollout restart deployment/postgres -n vitalstream
   ```

3. **Implement Connection Pooling Middleware**
   ```bash
   # Add connection pool monitoring
   kubectl patch deployment vitalstream-backend -n vitalstream -p '{
     "spec": {
       "template": {
         "spec": {
           "containers": [{
             "name": "backend",
             "env": [
               {"name": "DB_POOL_MONITOR", "value": "true"},
               {"name": "DB_POOL_TIMEOUT", "value": "30"},
               {"name": "DB_POOL_RECYCLE", "value": "3600"}
             ]
           }]
         }
       }
     }
   }'
   ```

### Step 3: Query Optimization (15-30 minutes)
1. **Kill Long-Running Queries**
   ```bash
   # Identify and kill problematic queries
   kubectl exec -it deployment/postgres -n vitalstream -- psql -U postgres -d vitalstream -c "
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE (now() - pg_stat_activity.query_start) > interval '10 minutes')
   AND state = 'active';
   "
   ```

2. **Optimize Slow Queries**
   ```bash
   # Check query performance
   kubectl exec -it deployment/postgres -n vitalstream -- psql -U postgres -d vitalstream -c "
   EXPLAIN ANALYZE SELECT * FROM patients WHERE created_at > NOW() - INTERVAL '1 day';
   "
   
   # Add indexes if needed
   kubectl exec -it deployment/postgres -n vitalstream -- psql -U postgres -d vitalstream -c "
   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_patients_created_at ON patients(created_at);
   "
   ```

3. **Implement Query Caching**
   ```bash
   # Enable query result caching
   kubectl exec -it deployment/postgres -n vitalstream -- psql -U postgres -d vitalstream -c "
   ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
   SELECT pg_reload_conf();
   "
   ```

### Step 4: Database Scaling (30+ minutes)
1. **Scale Database Resources**
   ```bash
   # Increase database resources
   kubectl patch deployment postgres -n vitalstream -p '{
     "spec": {
       "template": {
         "spec": {
           "containers": [{
             "name": "postgres",
             "resources": {
               "limits": {
                 "cpu": "4000m",
                 "memory": "8Gi"
               },
               "requests": {
                 "cpu": "2000m",
                 "memory": "4Gi"
               }
             }
           }]
         }
       }
     }
   }'
   ```

2. **Implement Read Replicas**
   ```bash
   # Deploy read replica
   kubectl apply -f - <<EOF
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: postgres-replica
     namespace: vitalstream
   spec:
     replicas: 1
     selector:
       matchLabels:
         app: postgres-replica
     template:
       metadata:
         labels:
           app: postgres-replica
       spec:
         containers:
         - name: postgres-replica
           image: postgres:16-alpine
           env:
           - name: POSTGRES_REPLICATION_MODE
             value: "slave"
           - name: POSTGRES_MASTER_SERVICE
             value: "postgres"
           ports:
           - containerPort: 5432
   EOF
   ```

3. **Implement Connection Pooling Service**
   ```bash
   # Deploy PgBouncer for connection pooling
   kubectl apply -f - <<EOF
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: pgbouncer
     namespace: vitalstream
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: pgbouncer
     template:
       metadata:
         labels:
           app: pgbouncer
       spec:
         containers:
         - name: pgbouncer
           image: pgbouncer/pgbouncer:latest
           ports:
           - containerPort: 6432
           env:
           - name: DATABASES_HOST
             value: "postgres"
           - name: DATABASES_PORT
             value: "5432"
           - name: DATABASES_USER
             value: "postgres"
           - name: DATABASES_PASSWORD
             value: "password"
           - name: DATABASES_DBNAME
             value: "vitalstream"
   EOF
   ```

## Prevention

### 1. Connection Pool Monitoring
- Implement real-time connection pool monitoring
- Set up connection leak detection
- Add connection usage alerts
- Implement automated connection cleanup

### 2. Database Performance
- Regular query performance analysis
- Implement query optimization
- Add database performance monitoring
- Implement automated indexing

### 3. Capacity Planning
- Regular connection capacity planning
- Implement auto-scaling for database
- Add resource usage monitoring
- Plan for peak loads

### 4. Application Optimization
- Implement proper connection management
- Add connection timeout handling
- Implement retry mechanisms
- Add circuit breakers

## Escalation

### Level 1 Escalation (After 15 minutes)
- **Contact**: Database Team
- **Slack**: #database-team
- **Email**: database-team@vitalstream.com
- **Phone**: +1-555-DB-TEAM

### Level 2 Escalation (After 30 minutes)
- **Contact**: Infrastructure Lead
- **Slack**: @infra-lead
- **Email**: infra-lead@vitalstream.com
- **Phone**: +1-555-INFRA-LEAD

### Level 3 Escalation (After 60 minutes)
- **Contact**: VP of Engineering
- **Slack**: @vp-engineering
- **Email**: vp-eng@vitalstream.com
- **Phone**: +1-555-VP-ENG

## Post-Incident Actions

1. **Database Analysis**
   - Analyze connection patterns
   - Identify performance bottlenecks
   - Document root causes
   - Update monitoring

2. **Performance Optimization**
   - Optimize slow queries
   - Add missing indexes
   - Update connection pooling
   - Implement caching

3. **Infrastructure Improvements**
   - Scale database resources
   - Implement read replicas
   - Add connection pooling service
   - Update monitoring

4. **Process Improvements**
   - Update connection management procedures
   - Improve incident response
   - Add automated recovery
   - Update documentation

## Related Documentation
- [Database Architecture](https://docs.vitalstream.com/database)
- [Connection Pooling Guide](https://docs.vitalstream.com/connection-pooling)
- [Performance Tuning](https://docs.vitalstream.com/database-performance)
- [PostgreSQL Best Practices](https://docs.vitalstream.com/postgresql-best-practices)

## Tools and Commands Reference

### Quick Commands
```bash
# Check active connections
kubectl exec -it deployment/postgres -n vitalstream -- psql -U postgres -d vitalstream -c "SELECT count(*) FROM pg_stat_activity;"

# Check connection pool usage
curl -s "http://prometheus:9090/api/v1/query?query=database_connections_active{environment=\"production\"} / database_connections_max{environment=\"production\"} * 100"

# Kill long-running queries
kubectl exec -it deployment/postgres -n vitalstream -- psql -U postgres -d vitalstream -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE (now() - query_start) > interval '10 minutes';"

# Restart database
kubectl rollout restart deployment/postgres -n vitalstream

# Scale database
kubectl scale deployment/postgres --replicas=2 -n vitalstream
```

### Monitoring Queries
```promql
# Database connection pool usage
database_connections_active{environment="production"} / database_connections_max{environment="production"} * 100

# Database query duration
histogram_quantile(0.95, rate(database_query_duration_seconds_bucket{environment="production"}[5m]))

# Database error rate
rate(database_errors_total{environment="production"}[5m])

# Database resource usage
rate(container_cpu_usage_seconds_total{container="postgres",environment="production"}[5m]) * 100
```

---

**Last Updated**: 2026-01-16
**Version**: 1.0
**Maintainer**: Database Team
**Review Date**: 2026-04-16
