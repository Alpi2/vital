# Runbook: ECG Processing High Latency

## Severity
**WARNING**

## Description
ECG processing latency has exceeded the 1-second threshold for p95 response time. This indicates that ECG segments are taking too long to process, which can affect real-time monitoring and clinical decision support.

## Impact
- **Patient Monitoring**: Delayed ECG analysis results
- **Clinical Decisions**: Slower anomaly detection and alerts
- **User Experience**: Delayed updates in dashboards
- **System Performance**: Potential backlog in processing queue

## Diagnosis

### 1. Check Grafana Dashboard
- **Dashboard**: [ECG Processing Dashboard](https://grafana.vitalstream.com/d/vitalstream-ecg-pipeline)
- **Key Metrics to Check**:
  - ECG Processing Duration (p95 should be <1s)
  - Processing Queue Depth
  - R-Peak Detection Time
  - Feature Extraction Time
  - ML Inference Time

### 2. Check Prometheus Metrics
```bash
# Check ECG processing latency by stage
curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(ecg_processing_duration_seconds_bucket{environment=\"production\"}[5m]))"

# Check processing queue depth
curl -s "http://prometheus:9090/api/v1/query?query=ecg_processing_queue_depth"

# Check error rates
curl -s "http://prometheus:9090/api/v1/query?query=rate(ecg_processing_errors_total{environment=\"production\"}[5m])"
```

### 3. Check Service Health
```bash
# Check ECG processing service
curl -f http://ecg-processing:8001/health || echo "ECG processing service down"

# Check detailed health
curl -f http://ecg-processing:8001/health/detailed || echo "Detailed health check failed"

# Check service metrics
curl -s http://ecg-processing:8001/metrics | grep ecg_processing
```

### 4. Check Resource Usage
```bash
# Check ECG processing pod resources
kubectl top pods -l app=ecg-processing -n vitalstream

# Check resource limits
kubectl describe pod -l app=ecg-processing -n vitalstream

# Check node resources
kubectl top nodes
```

### 5. Check Processing Queue
```bash
# Check queue status
kubectl exec -it deployment/ecg-processing -n vitalstream -- python -c "
import redis
r = redis.Redis(host='redis', port=6379)
queue_size = r.llen('ecg_processing_queue')
print(f'Queue size: {queue_size}')
"
```

## Resolution Steps

### Step 1: Immediate Triage (First 5 minutes)
1. **Verify Alert Accuracy**
   ```bash
   # Check current latency
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(ecg_processing_duration_seconds_bucket{environment=\"production\"}[2m]))"
   ```

2. **Identify Bottleneck Stage**
   ```bash
   # Check latency by stage
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(ecg_processing_duration_seconds_bucket{environment=\"production\",stage!=\"\"}[2m])) by (stage)"
   ```

3. **Check Queue Status**
   ```bash
   # Check if queue is backing up
   kubectl exec -it deployment/ecg-processing -n vitalstream -- python -c "
   import redis
   r = redis.Redis(host='redis', port=6379)
   print(f'Queue size: {r.llen(\"ecg_processing_queue\")}')
   print(f'Processing rate: {r.llen(\"ecg_processing_queue\") / 60:.2f} items/sec')
   "
   ```

### Step 2: Service Optimization (5-15 minutes)
1. **Scale Up ECG Processing**
   ```bash
   # Increase replica count
   kubectl scale deployment/ecg-processing --replicas=3 -n vitalstream
   
   # Check scaling status
   kubectl rollout status deployment/ecg-processing -n vitalstream
   ```

2. **Increase Resource Limits**
   ```bash
   # Patch with higher CPU limits
   kubectl patch deployment ecg-processing -n vitalstream -p '{
     "spec": {
       "template": {
         "spec": {
           "containers": [{
             "name": "ecg-processing",
             "resources": {
               "limits": {
                 "cpu": "3000m",
                 "memory": "6Gi"
               },
               "requests": {
                 "cpu": "1500m",
                 "memory": "3Gi"
               }
             }
           }]
         }
       }
     }
   }'
   ```

3. **Optimize Processing Parameters**
   ```bash
   # Check current configuration
   kubectl exec -it deployment/ecg-processing -n vitalstream -- cat /app/config/ecg_config.yaml
   
   # Update configuration for better performance
   kubectl patch configmap ecg-config -n vitalstream -p '{
     "data": {
       "ecg_config.yaml": "batch_size: 50\nmax_workers: 8\nprocessing_timeout: 30"
     }
   }'
   
   # Restart to apply changes
   kubectl rollout restart deployment/ecg-processing -n vitalstream
   ```

### Step 3: Advanced Optimization (15-30 minutes)
1. **Check WASM Performance**
   ```bash
   # Check WASM module performance
   kubectl exec -it deployment/ecg-processing -n vitalstream -- python -c "
   import time
   from ecg_wasm import process_ecg_segment
   
   # Test with sample data
   test_data = [0.1] * 1000  # Sample ECG data
   start_time = time.time()
   result = process_ecg_segment(test_data)
   processing_time = time.time() - start_time
   print(f'WASM processing time: {processing_time:.4f}s')
   "
   ```

2. **Optimize Database Queries**
   ```bash
   # Check slow queries
   kubectl exec -it deployment/postgres -n vitalstream -- psql -U postgres -d vitalstream -c "
   SELECT query, mean_time, calls, total_time
   FROM pg_stat_statements
   WHERE mean_time > 1000
   ORDER BY mean_time DESC
   LIMIT 10;
   "
   ```

3. **Implement Caching**
   ```bash
   # Check cache hit rate
   curl -s "http://prometheus:9090/api/v1/query?query=cache_hit_rate{cache_type=\"ecg_processing\"}"
   
   # If hit rate is low, consider increasing cache size
   kubectl patch configmap redis-config -n vitalstream -p '{
     "data": {
       "redis.conf": "maxmemory 2gb\\nmaxmemory-policy allkeys-lru"
     }
   }'
   ```

### Step 4: Load Balancing (30+ minutes)
1. **Implement Horizontal Pod Autoscaler**
   ```bash
   # Create HPA for ECG processing
   kubectl autoscale deployment ecg-processing \
     --cpu-percent=70 \
     --min=2 \
     --max=10 \
     -n vitalstream
   ```

2. **Implement Queue-Based Processing**
   ```bash
   # Check if queue worker is properly configured
   kubectl get deployment ecg-queue-worker -n vitalstream
   
   # Scale queue workers
   kubectl scale deployment ecg-queue-worker --replicas=5 -n vitalstream
   ```

3. **Implement Circuit Breaker**
   ```bash
   # Check if circuit breaker is configured
   kubectl get configmap ecg-config -n vitalstream -o yaml | grep circuit_breaker
   
   # Configure circuit breaker if not present
   kubectl patch configmap ecg-config -n vitalstream -p '{
     "data": {
       "circuit_breaker.yaml": "failure_threshold: 5\nrecovery_timeout: 30\nhalf_open_max_calls: 3"
     }
   }'
   ```

## Prevention

### 1. Performance Monitoring
- Implement real-time latency monitoring
- Set up predictive alerting for latency trends
- Add performance regression testing
- Implement automated performance testing

### 2. Capacity Planning
- Regular load testing
- Implement auto-scaling policies
- Monitor resource utilization trends
- Plan for peak loads

### 3. Code Optimization
- Profile ECG processing code
- Optimize WASM modules
- Implement efficient data structures
- Use connection pooling

### 4. Infrastructure Optimization
- Use faster storage (SSD)
- Implement edge caching
- Optimize network latency
- Use dedicated resources for ECG processing

## Escalation

### Level 1 Escalation (After 15 minutes)
- **Contact**: ECG Processing Team
- **Slack**: #ecg-team
- **Email**: ecg-team@vitalstream.com
- **Phone**: +1-555-ECG-TEAM

### Level 2 Escalation (After 30 minutes)
- **Contact**: Performance Engineering Team
- **Slack**: #perf-team
- **Email**: perf-team@vitalstream.com
- **Phone**: +1-555-PERF-TEAM

### Level 3 Escalation (After 60 minutes)
- **Contact**: Infrastructure Lead
- **Slack**: @infra-lead
- **Email**: infra-lead@vitalstream.com
- **Phone**: +1-555-INFRA-LEAD

## Post-Incident Actions

1. **Performance Analysis**
   - Analyze latency patterns
   - Identify bottlenecks
   - Document root causes

2. **Optimization Implementation**
   - Implement identified optimizations
   - Update configuration
   - Add monitoring gaps

3. **Testing**
   - Load test optimizations
   - Validate performance improvements
   - Update performance baselines

4. **Documentation**
   - Update runbook with new steps
   - Document configuration changes
   - Share lessons learned

## Related Documentation
- [ECG Processing Architecture](https://docs.vitalstream.com/ecg-processing)
- [Performance Tuning Guide](https://docs.vitalstream.com/performance-tuning)
- [WASM Optimization](https://docs.vitalstream.com/wasm-optimization)
- [Auto-Scaling Guide](https://docs.vitalstream.com/auto-scaling)

## Tools and Commands Reference

### Quick Commands
```bash
# Check ECG processing latency
curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(ecg_processing_duration_seconds_bucket{environment=\"production\"}[5m]))"

# Scale ECG processing
kubectl scale deployment/ecg-processing --replicas=3 -n vitalstream

# Check queue status
kubectl exec -it deployment/ecg-processing -n vitalstream -- python -c "
import redis
r = redis.Redis(host='redis', port=6379)
print(f'Queue size: {r.llen(\"ecg_processing_queue\")}')
"

# Check resource usage
kubectl top pods -l app=ecg-processing -n vitalstream

# Check logs
kubectl logs -f -l app=ecg-processing -n vitalstream
```

### Monitoring Queries
```promql
# ECG processing latency by stage
histogram_quantile(0.95, rate(ecg_processing_duration_seconds_bucket{environment="production",stage!=""}[5m])) by (stage)

# Processing queue depth
ecg_processing_queue_depth

# Error rate by type
rate(ecg_processing_errors_total{environment="production"}[5m]) by (error_type)

# Processing throughput
rate(ecg_processing_total{status="success",environment="production"}[5m])

# Resource usage
rate(container_cpu_usage_seconds_total{pod=~"ecg-processing.*",environment="production"}[5m]) * 100
```

---

**Last Updated**: 2026-01-16
**Version**: 1.0
**Maintainer**: ECG Processing Team
**Review Date**: 2026-04-16
