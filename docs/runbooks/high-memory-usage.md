# Runbook: High Memory Usage

## Severity
**WARNING / CRITICAL**

## Description
System is experiencing high memory usage that can lead to performance degradation, service instability, or out-of-memory errors. This affects application performance and can cause service crashes.

## Impact
- **Application Performance**: Slow response times and degraded performance
- **Service Stability**: Potential crashes or restarts due to OOM
- **User Experience**: Sluggish interface and timeouts
- **System Health**: Overall system degradation

## Diagnosis

### 1. Check Grafana Dashboard
- **Dashboard**: [System Health Dashboard](https://grafana.vitalstream.com/d/vitalstream-system-health)
- **Key Metrics to Check**:
  - Memory Usage by Container
  - Memory Usage by Node
  - OOM Events
  - Swap Usage

### 2. Check Prometheus Metrics
```bash
# Check memory usage by container
curl -s "http://prometheus:9090/api/v1/query?query=(container_memory_usage_bytes{environment=\"production\"} / container_spec_memory_limit_bytes) * 100"

# Check memory usage by node
curl -s "http://prometheus:9090/api/v1/query?query=(node_memory_MemTotal_bytes{environment=\"production\"} - node_memory_MemAvailable_bytes{environment=\"production\"}) / node_memory_MemTotal_bytes{environment=\"production\"} * 100"

# Check OOM events
curl -s "http://prometheus:9090/api/v1/query?query(rate(container_oom_events_total{environment=\"production\"}[5m]))"
```

### 3. Check System Memory
```bash
# Check memory usage on nodes
kubectl top nodes

# Check memory usage by pod
kubectl top pods -n vitalstream --sort-by=memory

# Check detailed memory usage
kubectl exec -it deployment/vitalstream-backend -n vitalstream -- free -h

# Check memory pressure
kubectl exec -it deployment/vitalstream-backend -n vitalstream -- cat /proc/meminfo | grep -E "(MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree)"
```

### 4. Check Application Memory Usage
```bash
# Check Python memory usage
kubectl exec -it deployment/vitalstream-backend -n vitalstream -- python -c "
import psutil
import os
process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f'RSS: {memory_info.rss / 1024 / 1024:.2f} MB')
print(f'VMS: {memory_info.vms / 1024 / 1024:.2f} MB')
print(f'Memory percent: {process.memory_percent():.2f}%')
"

# Check Java memory usage (if applicable)
kubectl exec -it deployment/ml-inference -n vitalstream -- jstat -gc 1

# Check Node.js memory usage (if applicable)
kubectl exec -it deployment/websocket -n vitalstream -- node -e "
console.log('Memory usage:', process.memoryUsage());
console.log('Heap used:', process.memoryUsage().heapUsed / 1024 / 1024, 'MB');
"
```

### 5. Check Memory Leaks
```bash
# Check for memory leaks in Python
kubectl exec -it deployment/vitalstream-backend -n vitalstream -- python -c "
import gc
import objgraph
import time

# Check object counts
print('Object counts:')
print(f'dict: {len([obj for obj in gc.get_objects() if type(obj) is dict])}')
print(f'list: {len([obj for obj in gc.get_objects() if type(obj) is list])}')
print(f'tuple: {len([obj for obj in gc.get_objects() if type(obj) is tuple])}')

# Force garbage collection
gc.collect()
print('After GC:')
print(f'dict: {len([obj for obj in gc.get_objects() if type(obj) is dict])}')
print(f'list: {len([obj for obj in gc.get_objects() if type(obj) is list])}')
"
```

## Resolution Steps

### Step 1: Immediate Triage (First 5 minutes)
1. **Verify Alert Accuracy**
   ```bash
   # Check current memory usage
   kubectl top pods -n vitalstream --sort-by=memory
   ```

2. **Identify Memory-Hungry Pods**
   ```bash
   # Find pods using most memory
   kubectl top pods -n vitalstream | sort -k4 -nr | head -10
   
   # Check pod details
   kubectl describe pod <pod-name> -n vitalstream
   ```

3. **Check for OOM Events**
   ```bash
   # Check OOM killed pods
   kubectl get events -n vitalstream --field-selector reason=OOMKilling
   
   # Check pod restarts
   kubectl get pods -n vitalstream | grep -E "(RESTARTING|CrashLoopBackOff)"
   ```

### Step 2: Memory Optimization (5-15 minutes)
1. **Increase Memory Limits**
   ```bash
   # Increase memory limits for problematic pods
   kubectl patch deployment vitalstream-backend -n vitalstream -p '{
     "spec": {
       "template": {
         "spec": {
           "containers": [{
             "name": "backend",
             "resources": {
               "limits": {
                 "memory": "4Gi"
               },
               "requests": {
                 "memory": "2Gi"
               }
             }
           }]
         }
       }
     }
   }'
   ```

2. **Restart Memory-Hungry Services**
   ```bash
   # Restart services with high memory usage
   kubectl rollout restart deployment/vitalstream-backend -n vitalstream
   kubectl rollout restart deployment/ml-inference -n vitalstream
   kubectl rollout restart deployment/websocket -n vitalstream
   ```

3. **Enable Memory Profiling**
   ```bash
   # Add memory profiling to Python applications
   kubectl patch deployment vitalstream-backend -n vitalstream -p '{
     "spec": {
       "template": {
         "spec": {
           "containers": [{
             "name": "backend",
             "env": [
               {"name": "PYTHONTRACEMALLOC", "value": "1"},
               {"name": "PYTHONFAULTHANDLER", "value": "1"}
             ]
           }]
         }
       }
     }
   }'
   ```

### Step 3: Application Optimization (15-30 minutes)
1. **Optimize Python Memory Usage**
   ```bash
   # Add memory optimization settings
   kubectl patch deployment vitalstream-backend -n vitalstream -p '{
     "spec": {
       "template": {
         "spec": {
           "containers": [{
             "name": "backend",
             "env": [
               {"name": "PYTHONHASHSEED", "value": "random"},
               {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
               {"name": "PYTHONUNBUFFERED", "value": "1"}
             ]
           }]
         }
       }
     }
   }'
   ```

2. **Implement Memory Caching**
   ```bash
   # Add Redis for caching
   kubectl apply -f - <<EOF
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: redis-cache
     namespace: vitalstream
   spec:
     replicas: 1
     selector:
       matchLabels:
         app: redis-cache
     template:
       metadata:
         labels:
           app: redis-cache
       spec:
         containers:
         - name: redis
           image: redis:7.2-alpine
           ports:
           - containerPort: 6379
           command: ["redis-server", "--maxmemory", "1gb", "--maxmemory-policy", "allkeys-lru"]
   EOF
   ```

3. **Optimize Database Connections**
   ```bash
   # Reduce database connection pool size
   kubectl patch configmap backend-config -n vitalstream -p '{
     "data": {
       "database.yaml": "max_connections: 20\\nmin_connections: 5\\nconnection_timeout: 30"
     }
   }'
   ```

### Step 4: System-Level Optimization (30+ minutes)
1. **Add Swap Space**
   ```bash
   # Add swap to nodes (if not already present)
   kubectl apply -f - <<EOF
   apiVersion: v1
   kind: DaemonSet
   metadata:
     name: swap-init
     namespace: kube-system
   spec:
     selector:
       matchLabels:
         name: swap-init
     template:
       metadata:
         labels:
           name: swap-init
       spec:
         containers:
         - name: swap-init
           image: busybox
           command: ['sh', '-c', 'fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile']
           volumeMounts:
           - name: root
             mountPath: /host
           securityContext:
             privileged: true
         volumes:
         - name: root
           hostPath:
             path: /
   EOF
   ```

2. **Implement Memory Limits**
   ```bash
   # Set memory limits at namespace level
   kubectl apply -f - <<EOF
   apiVersion: v1
   kind: LimitRange
   metadata:
     name: memory-limits
     namespace: vitalstream
   spec:
     limits:
     - default:
         memory: "2Gi"
       defaultRequest:
         memory: "1Gi"
       type: Container
   EOF
   ```

3. **Enable Memory Pressure Detection**
   ```bash
   # Add memory pressure monitoring
   kubectl apply -f - <<EOF
   apiVersion: v1
   kind: Pod
   metadata:
     name: memory-monitor
     namespace: vitalstream
   spec:
     containers:
     - name: memory-monitor
       image: prom/node-exporter:latest
       command:
       - /bin/sh
       - -c
       - |
         while true; do
           echo "Memory usage: $(free | grep Mem | awk '{print $3/$2 * 100.0}')"
           sleep 30
         done
       resources:
         requests:
           memory: "128Mi"
           cpu: "100m"
   EOF
   ```

## Prevention

### 1. Memory Monitoring
- Implement real-time memory usage monitoring
- Set up memory leak detection
- Add memory usage trend analysis
- Implement automated memory optimization

### 2. Application Optimization
- Implement proper memory management
- Add memory profiling tools
- Optimize data structures
- Implement memory-efficient algorithms

### 3. Resource Management
- Implement proper resource limits
- Add memory usage quotas
- Implement auto-scaling based on memory
- Add resource usage alerts

### 4. System Optimization
- Optimize system memory settings
- Implement memory compression
- Add swap space management
- Optimize garbage collection

## Escalation

### Level 1 Escalation (After 15 minutes)
- **Contact**: Infrastructure Team
- **Slack**: #infra-team
- **Email**: infra-team@vitalstream.com
- **Phone**: +1-555-INFRA-TEAM

### Level 2 Escalation (After 30 minutes)
- **Contact**: Platform Engineering Lead
- **Slack**: @platform-lead
- **Email**: platform-lead@vitalstream.com
- **Phone**: +1-555-PLATFORM

### Level 3 Escalation (After 60 minutes)
- **Contact**: VP of Engineering
- **Slack**: @vp-engineering
- **Email**: vp-eng@vitalstream.com
- **Phone**: +1-555-VP-ENG

## Post-Incident Actions

1. **Memory Analysis**
   - Analyze memory usage patterns
   - Identify memory leaks
   - Document root causes
   - Update monitoring

2. **Application Optimization**
   - Optimize memory usage
   - Fix memory leaks
   - Implement memory profiling
   - Update best practices

3. **Infrastructure Improvements**
   - Optimize resource allocation
   - Implement memory limits
   - Add memory monitoring
   - Update scaling policies

4. **Process Improvements**
   - Update memory management procedures
   - Improve incident response
   - Add automated recovery
   - Update documentation

## Related Documentation
- [Memory Management Guide](https://docs.vitalstream.com/memory-management)
- [Resource Optimization](https://docs.vitalstream.com/resource-optimization)
- [Performance Tuning](https://docs.vitalstream.com/performance-tuning)
- [Kubernetes Resource Management](https://docs.vitalstream.com/k8s-resources)

## Tools and Commands Reference

### Quick Commands
```bash
# Check memory usage by pod
kubectl top pods -n vitalstream --sort-by=memory

# Check memory usage by node
kubectl top nodes

# Check detailed memory usage
kubectl exec -it <pod-name> -n vitalstream -- free -h

# Check OOM events
kubectl get events -n vitalstream --field-selector reason=OOMKilling

# Restart memory-hungry pods
kubectl rollout restart deployment/<deployment-name> -n vitalstream

# Increase memory limits
kubectl patch deployment <deployment-name> -n vitalstream -p '{"spec":{"template":{"spec":{"containers":[{"name":"<container-name>","resources":{"limits":{"memory":"4Gi"}}}]}}}}'
```

### Monitoring Queries
```promql
# Memory usage by container
(container_memory_usage_bytes{environment="production"} / container_spec_memory_limit_bytes) * 100

# Memory usage by node
(node_memory_MemTotal_bytes{environment="production"} - node_memory_MemAvailable_bytes{environment="production"}) / node_memory_MemTotal_bytes{environment="production"} * 100

# OOM events
rate(container_oom_events_total{environment="production"}[5m])

# Swap usage
(node_memory_SwapTotal_bytes{environment="production"} - node_memory_SwapFree_bytes{environment="production"}) / node_memory_SwapTotal_bytes{environment="production"} * 100
```

---

**Last Updated**: 2026-01-16
**Version**: 1.0
**Maintainer**: Infrastructure Team
**Review Date**: 2026-04-16
