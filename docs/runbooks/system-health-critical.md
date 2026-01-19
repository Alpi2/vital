# Runbook: System Health Critical

## Severity
**CRITICAL**

## Description
System health score has dropped below 50, indicating severe degradation of VitalStream services. This is a critical incident that requires immediate attention as it affects patient monitoring and ECG processing capabilities.

## Impact
- **Patient Safety**: Real-time monitoring may be compromised
- **Clinical Decision Support**: ML inference and anomaly detection may be unavailable
- **Provider Experience**: Dashboard and report generation may be slow or unavailable
- **Data Loss**: Potential loss of ECG data if processing pipeline is affected

## Diagnosis

### 1. Check Grafana Dashboard
- **Dashboard**: [System Health Dashboard](https://grafana.vitalstream.com/d/vitalstream-system-health)
- **Key Metrics to Check**:
  - System Health Score (should be >80)
  - Service Availability (should be >99.9%)
  - Active Patients (monitor for sudden drops)
  - Critical Alerts (check for active alerts)

### 2. Check Prometheus Metrics
```bash
# Check system health score
curl -s "http://prometheus:9090/api/v1/query?query=avg(system_health_score{environment=\"production\"})"

# Check service availability
curl -s "http://prometheus:9090/api/v1/query?query=avg(service_availability{environment=\"production\"})"

# Check error rates
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(http_requests_total{status=~\"5..\",environment=\"production\"}[5m]))"
```

### 3. Check Service Status
```bash
# Check backend service
curl -f http://backend:8000/health || echo "Backend service down"

# Check ECG processing
curl -f http://ecg-processing:8001/health || echo "ECG processing down"

# Check ML inference
curl -f http://ml-inference:8002/health || echo "ML inference down"

# Check WebSocket service
curl -f http://websocket:8003/health || echo "WebSocket service down"
```

### 4. Check Recent Deployments
```bash
# Check recent deployments
kubectl get deployments --sort-by=.metadata.creationTimestamp -n vitalstream

# Check deployment status
kubectl rollout status deployment/vitalstream-backend -n vitalstream
kubectl rollout status deployment/ecg-processing -n vitalstream
kubectl rollout status deployment/ml-inference -n vitalstream
```

### 5. Check Resource Usage
```bash
# Check CPU and memory usage
kubectl top pods -n vitalstream

# Check node resources
kubectl top nodes

# Check resource limits
kubectl describe pod -l app=vitalstream-backend -n vitalstream
```

## Resolution Steps

### Step 1: Immediate Triage (First 5 minutes)
1. **Verify Alert Accuracy**
   ```bash
   # Check if metrics are being reported
   curl -s "http://prometheus:9090/api/v1/query?query=up{job=\"vitalstream-backend\"}"
   ```

2. **Identify Affected Services**
   ```bash
   # Check which services are down
   for service in backend ecg-processing ml-inference websocket; do
     if ! curl -f http://$service:8000/health; then
       echo "Service $service is down"
     fi
   done
   ```

3. **Check for Recent Changes**
   ```bash
   # Check recent commits
   git log --oneline -10
   # Check recent deployments
   kubectl get events --sort-by=.lastTimestamp -n vitalstream | tail -20
   ```

### Step 2: Service Recovery (5-15 minutes)
1. **Restart Affected Services**
   ```bash
   # Restart backend service
   kubectl rollout restart deployment/vitalstream-backend -n vitalstream
   
   # Restart ECG processing
   kubectl rollout restart deployment/ecg-processing -n vitalstream
   
   # Restart ML inference
   kubectl rollout restart deployment/ml-inference -n vitalstream
   ```

2. **Check Database Connectivity**
   ```bash
   # Test database connection
   kubectl exec -it deployment/vitalstream-backend -n vitalstream -- python -c "
   import psycopg2
   try:
       conn = psycopg2.connect('postgresql://user:pass@postgres:5432/vitalstream')
       print('Database connection successful')
   except Exception as e:
       print(f'Database connection failed: {e}')
   "
   ```

3. **Check Redis Connectivity**
   ```bash
   # Test Redis connection
   kubectl exec -it deployment/vitalstream-backend -n vitalstream -- python -c "
   import redis
   try:
       r = redis.Redis(host='redis', port=6379)
       r.ping()
       print('Redis connection successful')
   except Exception as e:
       print(f'Redis connection failed: {e}')
   "
   ```

### Step 3: Resource Scaling (15-30 minutes)
1. **Scale Up Resources**
   ```bash
   # Increase replica count
   kubectl scale deployment/vitalstream-backend --replicas=3 -n vitalstream
   kubectl scale deployment/ecg-processing --replicas=2 -n vitalstream
   kubectl scale deployment/ml-inference --replicas=2 -n vitalstream
   ```

2. **Increase Resource Limits**
   ```bash
   # Patch deployment with higher limits
   kubectl patch deployment vitalstream-backend -n vitalstream -p '{
     "spec": {
       "template": {
         "spec": {
           "containers": [{
             "name": "backend",
             "resources": {
               "limits": {
                 "cpu": "2000m",
                 "memory": "4Gi"
               },
               "requests": {
                 "cpu": "1000m",
                 "memory": "2Gi"
               }
             }
           }]
         }
       }
     }
   }'
   ```

### Step 4: Advanced Troubleshooting (30+ minutes)
1. **Check Network Connectivity**
   ```bash
   # Test service-to-service communication
   kubectl exec -it deployment/vitalstream-backend -n vitalstream -- \
     curl -f http://ecg-processing:8001/health
   
   kubectl exec -it deployment/ecg-processing -n vitalstream -- \
     curl -f http://ml-inference:8002/health
   ```

2. **Check Persistent Storage**
   ```bash
   # Check PVC status
   kubectl get pvc -n vitalstream
   
   # Check storage class
   kubectl get storageclass
   ```

3. **Check External Dependencies**
   ```bash
   # Test external API connectivity
   kubectl exec -it deployment/vitalstream-backend -n vitalstream -- \
     curl -f https://api.external-service.com/health
   ```

## Prevention

### 1. Monitoring Improvements
- Add more granular health checks
- Implement synthetic monitoring
- Set up predictive alerting
- Add more SLOs with appropriate thresholds

### 2. Infrastructure Improvements
- Implement auto-scaling policies
- Add circuit breakers for external services
- Implement graceful degradation
- Add more redundancy

### 3. Process Improvements
- Implement canary deployments
- Add more comprehensive testing
- Improve incident response procedures
- Regular chaos engineering exercises

## Escalation

### Level 1 Escalation (After 15 minutes)
- **Contact**: On-call SRE team
- **Slack**: #sre-team
- **Email**: sre-team@vitalstream.com
- **Phone**: +1-555-SRE-ALERT

### Level 2 Escalation (After 30 minutes)
- **Contact**: Engineering Manager
- **Slack**: @engineering-manager
- **Email**: eng-manager@vitalstream.com
- **Phone**: +1-555-ENG-MANAGER

### Level 3 Escalation (After 60 minutes)
- **Contact**: VP of Engineering
- **Slack**: @vp-engineering
- **Email**: vp-eng@vitalstream.com
- **Phone**: +1-555-VP-ENG

## Post-Incident Actions

1. **Create Incident Report**
   - Timeline of events
   - Root cause analysis
   - Impact assessment
   - Resolution steps taken

2. **Update Runbooks**
   - Add new troubleshooting steps
   - Update contact information
   - Add lessons learned

3. **Implement Preventive Measures**
   - Fix identified issues
   - Add monitoring gaps
   - Improve documentation

4. **Team Debrief**
   - Schedule post-mortem meeting
   - Discuss what went well
   - Identify improvement opportunities

## Related Documentation
- [System Architecture](https://docs.vitalstream.com/architecture)
- [Monitoring Guide](https://docs.vitalstream.com/monitoring)
- [Incident Response](https://docs.vitalstream.com/incident-response)
- [SRE Playbook](https://docs.vitalstream.com/sre-playbook)

## Tools and Commands Reference

### Quick Commands
```bash
# Check all services health
kubectl get pods -n vitalstream -o wide

# Check logs for all services
kubectl logs -f -l app=vitalstream-backend -n vitalstream

# Check resource usage
kubectl top pods -n vitalstream --sort-by=cpu

# Check events
kubectl get events -n vitalstream --sort-by=.lastTimestamp

# Scale services
kubectl scale deployment/vitalstream-backend --replicas=3 -n vitalstream

# Restart services
kubectl rollout restart deployment/vitalstream-backend -n vitalstream
```

### Monitoring Queries
```promql
# System health score
avg(system_health_score{environment="production"})

# Service availability
avg(service_availability{environment="production"})

# Error rate
sum(rate(http_requests_total{status=~"5..",environment="production"}[5m]))

# CPU usage
rate(container_cpu_usage_seconds_total{environment="production"}[5m]) * 100

# Memory usage
(container_memory_usage_bytes{environment="production"} / container_spec_memory_limit_bytes) * 100
```

---

**Last Updated**: 2026-01-16
**Version**: 1.0
**Maintainer**: SRE Team
**Review Date**: 2026-04-16
