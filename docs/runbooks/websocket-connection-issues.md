# Runbook: WebSocket Connection Issues

## Severity
**WARNING / CRITICAL**

## Description
WebSocket service is experiencing connection issues that affect real-time ECG streaming and patient monitoring. This can impact the delivery of critical patient data to healthcare providers.

## Impact
- **Real-time Monitoring**: Delayed or interrupted ECG data streams
- **Patient Safety**: Potential loss of real-time cardiac monitoring
- **Provider Experience**: Disconnected dashboards and alerts
- **Data Integrity**: Potential data loss during connection issues

## Diagnosis

### 1. Check Grafana Dashboard
- **Dashboard**: [WebSocket Dashboard](https://grafana.vitalstream.com/d/vitalstream-websocket)
- **Key Metrics to Check**:
  - Active WebSocket Connections
  - Connection/Disconnection Rates
  - Message Latency
  - Connection Errors by Type

### 2. Check Prometheus Metrics
```bash
# Check active connections
curl -s "http://prometheus:9090/api/v1/query?query=sum(websocket_connections_active{environment=\"production\"})"

# Check disconnection rate
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(websocket_disconnections_total{environment=\"production\"}[5m])) by (reason)"

# Check message latency
curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(websocket_message_latency_seconds_bucket{environment=\"production\"}[5m]))"
```

### 3. Check Service Health
```bash
# Check WebSocket service
curl -f http://websocket:8003/health || echo "WebSocket service down"

# Check detailed health
curl -f http://websocket:8003/health/detailed || echo "Detailed health check failed"

# Check WebSocket endpoint
curl -I http://websocket:8003/ws || echo "WebSocket endpoint not responding"
```

### 4. Check Connection Logs
```bash
# Check WebSocket service logs
kubectl logs -f -l app=websocket -n vitalstream --tail=100

# Check connection events
kubectl logs -l app=websocket -n vitalstream | grep -E "(connect|disconnect|error)"

# Check nginx/proxy logs (if applicable)
kubectl logs -l app=nginx-ingress -n ingress-nginx | grep websocket
```

### 5. Test WebSocket Connection
```bash
# Test WebSocket connection
kubectl exec -it deployment/websocket -n vitalstream -- python -c "
import asyncio
import websockets
import json

async def test_connection():
    try:
        uri = 'ws://localhost:8003/ws'
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps({'type': 'ping'}))
            response = await websocket.recv()
            print(f'WebSocket response: {response}')
    except Exception as e:
        print(f'WebSocket connection failed: {e}')

asyncio.run(test_connection())
"
```

## Resolution Steps

### Step 1: Immediate Triage (First 5 minutes)
1. **Verify Alert Accuracy**
   ```bash
   # Check current disconnection rate
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(websocket_disconnections_total{environment=\"production\"}[2m]))"
   ```

2. **Check Service Status**
   ```bash
   # Check if service is running
   kubectl get pods -l app=websocket -n vitalstream
   
   # Check pod status
   kubectl describe pod -l app=websocket -n vitalstream
   ```

3. **Check Resource Usage**
   ```bash
   # Check WebSocket pod resources
   kubectl top pods -l app=websocket -n vitalstream
   
   # Check node resources
   kubectl top nodes
   ```

### Step 2: Service Recovery (5-15 minutes)
1. **Restart WebSocket Service**
   ```bash
   # Restart deployment
   kubectl rollout restart deployment/websocket -n vitalstream
   
   # Check rollout status
   kubectl rollout status deployment/websocket -n vitalstream
   ```

2. **Check Load Balancer Configuration**
   ```bash
   # Check service configuration
   kubectl get service websocket -n vitalstream -o yaml
   
   # Check ingress configuration
   kubectl get ingress websocket-ingress -n vitalstream -o yaml
   ```

3. **Verify Network Policies**
   ```bash
   # Check network policies
   kubectl get networkpolicy -n vitalstream
   
   # Test network connectivity
   kubectl exec -it deployment/websocket -n vitalstream -- nc -zv websocket 8003
   ```

### Step 3: Connection Optimization (15-30 minutes)
1. **Adjust WebSocket Configuration**
   ```bash
   # Check current configuration
   kubectl get configmap websocket-config -n vitalstream -o yaml
   
   # Update configuration for better connection handling
   kubectl patch configmap websocket-config -n vitalstream -p '{
     "data": {
       "websocket.yaml": "max_connections: 10000\\nheartbeat_interval: 30\\nconnection_timeout: 300\\nmax_message_size: 1048576"
     }
   }'
   
   # Restart to apply changes
   kubectl rollout restart deployment/websocket -n vitalstream
   ```

2. **Scale Up WebSocket Service**
   ```bash
   # Increase replica count
   kubectl scale deployment/websocket --replicas=5 -n vitalstream
   
   # Check scaling status
   kubectl rollout status deployment/websocket -n vitalstream
   ```

3. **Implement Connection Pooling**
   ```bash
   # Check if connection pooling is configured
   kubectl get configmap websocket-config -n vitalstream -o yaml | grep connection_pool
   
   # Configure connection pooling
   kubectl patch configmap websocket-config -n vitalstream -p '{
     "data": {
       "connection_pool.yaml": "max_pool_size: 1000\\npool_timeout: 30\\nmax_idle_time: 300"
     }
   }'
   ```

### Step 4: Advanced Troubleshooting (30+ minutes)
1. **Check SSL/TLS Configuration**
   ```bash
   # Check SSL certificate
   kubectl get secret websocket-tls -n vitalstream -o yaml
   
   # Test SSL connection
   kubectl exec -it deployment/websocket -n vitalstream -- openssl s_client -connect websocket:8003 -servername websocket.vitalstream.com
   ```

2. **Check Proxy Configuration**
   ```bash
   # Check nginx configuration
   kubectl exec -it deployment/nginx-ingress -n ingress-nginx -- cat /etc/nginx/nginx.conf | grep -A 10 websocket
   
   # Test proxy connection
   curl -I -H "Connection: Upgrade" -H "Upgrade: websocket" http://websocket.vitalstream.com/ws
   ```

3. **Implement Health Checks**
   ```bash
   # Add readiness and liveness probes
   kubectl patch deployment websocket -n vitalstream -p '{
     "spec": {
       "template": {
         "spec": {
           "containers": [{
             "name": "websocket",
             "readinessProbe": {
               "httpGet": {
                 "path": "/health",
                 "port": 8003
               },
               "initialDelaySeconds": 5,
               "periodSeconds": 10
             },
             "livenessProbe": {
               "httpGet": {
                 "path": "/health",
                 "port": 8003
               },
               "initialDelaySeconds": 15,
               "periodSeconds": 20
             }
           }]
         }
       }
     }
   }'
   ```

## Prevention

### 1. Connection Monitoring
- Implement real-time connection monitoring
- Set up connection pattern analysis
- Add connection anomaly detection
- Implement automated connection recovery

### 2. Load Balancing
- Implement proper WebSocket load balancing
- Add connection affinity
- Implement graceful connection migration
- Add connection draining

### 3. Resource Management
- Implement auto-scaling for WebSocket connections
- Add connection limit monitoring
- Implement resource usage alerts
- Add connection pooling

### 4. Network Optimization
- Optimize TCP settings for WebSocket
- Implement proper keep-alive settings
- Add network latency monitoring
- Implement connection compression

## Escalation

### Level 1 Escalation (After 15 minutes)
- **Contact**: WebSocket Team
- **Slack**: #websocket-team
- **Email**: websocket-team@vitalstream.com
- **Phone**: +1-555-WS-TEAM

### Level 2 Escalation (After 30 minutes)
- **Contact**: Infrastructure Team
- **Slack**: #infra-team
- **Email**: infra-team@vitalstream.com
- **Phone**: +1-555-INFRA-TEAM

### Level 3 Escalation (After 60 minutes)
- **Contact**: Platform Engineering Lead
- **Slack**: @platform-lead
- **Email**: platform-lead@vitalstream.com
- **Phone**: +1-555-PLATFORM

## Post-Incident Actions

1. **Connection Analysis**
   - Analyze connection patterns
   - Identify connection failure root causes
   - Document network issues
   - Update connection monitoring

2. **Infrastructure Improvements**
   - Optimize load balancer configuration
   - Improve network settings
   - Add connection redundancy
   - Update monitoring

3. **Process Improvements**
   - Update connection handling procedures
   - Improve incident response
   - Add automated recovery
   - Update documentation

4. **Team Debrief**
   - Schedule post-mortem meeting
   - Discuss connection issues
   - Identify improvement opportunities
   - Update best practices

## Related Documentation
- [WebSocket Architecture](https://docs.vitalstream.com/websocket)
- [Real-time Monitoring Guide](https://docs.vitalstream.com/real-time-monitoring)
- [Load Balancing Guide](https://docs.vitalstream.com/load-balancing)
- [Network Optimization](https://docs.vitalstream.com/network-optimization)

## Tools and Commands Reference

### Quick Commands
```bash
# Check WebSocket service health
curl -f http://websocket:8003/health

# Check active connections
curl -s "http://prometheus:9090/api/v1/query?query=sum(websocket_connections_active{environment=\"production\"})"

# Check disconnection rate
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(websocket_disconnections_total{environment=\"production\"}[5m])) by (reason)"

# Restart WebSocket service
kubectl rollout restart deployment/websocket -n vitalstream

# Scale WebSocket service
kubectl scale deployment/websocket --replicas=5 -n vitalstream

# Test WebSocket connection
kubectl exec -it deployment/websocket -n vitalstream -- python -c "
import asyncio
import websockets
async def test():
    try:
        async with websockets.connect('ws://localhost:8003/ws') as ws:
            print('Connected successfully')
    except Exception as e:
        print(f'Connection failed: {e}')
asyncio.run(test())
"
```

### Monitoring Queries
```promql
# Active WebSocket connections
sum(websocket_connections_active{environment="production"})

# Connection/disconnection rate
sum(rate(websocket_disconnections_total{environment="production"}[5m])) by (reason)

# Message latency
histogram_quantile(0.95, rate(websocket_message_latency_seconds_bucket{environment="production"}[5m]))

# Connection errors by type
sum(rate(websocket_connection_errors_total{environment="production"}[5m])) by (error_type)

# Resource usage
rate(container_cpu_usage_seconds_total{pod=~"websocket.*",environment="production"}[5m]) * 100
```

---

**Last Updated**: 2026-01-16
**Version**: 1.0
**Maintainer**: WebSocket Team
**Review Date**: 2026-04-16
