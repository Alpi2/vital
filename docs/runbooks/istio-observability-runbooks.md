# Istio Service Mesh Observability Runbooks

## 🚨 Critical Alert Response Procedures

### 1. High Error Rate Alert
**Alert**: `IstioHighErrorRate` - Error rate > 5% for 2 minutes

#### Immediate Actions (0-5 minutes)
1. **Identify affected services**
   ```bash
   # Check current error rates by service
   kubectl exec -n istio-system $(kubectl get pod -n istio-system -l app=prometheus -o jsonpath='{.items[0].metadata.name}') -- \
   promtool query instant 'sum(rate(istio_requests_total{response_code=~"5.."}[5m])) by (destination_service)'
   ```

2. **Check service health**
   ```bash
   # Verify pod status
   kubectl get pods -n vitalstream -o wide
   
   # Check recent events
   kubectl get events -n vitalstream --sort-by='.lastTimestamp' | tail -20
   ```

3. **Examine traces in Jaeger**
   - Navigate to Jaeger UI
   - Filter by affected service
   - Look for error spans and common patterns

#### Investigation Steps (5-30 minutes)
1. **Check upstream dependencies**
   ```bash
   # Test connectivity to upstream services
   kubectl exec -it deployment/backend -n vitalstream -- curl -v http://upstream-service/health
   ```

2. **Review recent deployments**
   ```bash
   # Check for recent changes
   kubectl rollout history deployment/backend -n vitalstream
   ```

3. **Analyze logs**
   ```bash
   # Check application logs
   kubectl logs -f deployment/backend -n vitalstream --tail=100
   ```

#### Resolution Actions
- If deployment issue: `kubectl rollout undo deployment/backend -n vitalstream`
- If resource constraints: Scale up or optimize resources
- If upstream dependency: Contact dependent team or implement circuit breaker

---

### 2. High Latency Alert
**Alert**: `IstioHighLatency` - P95 latency > 1000ms for 5 minutes

#### Immediate Actions
1. **Identify bottleneck services**
   ```bash
   # Check latency by service
   kubectl exec -n istio-system $(kubectl get pod -n istio-system -l app=prometheus -o jsonpath='{.items[0].metadata.name}') -- \
   promtool query instant 'histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket[5m])) by (le, destination_service))'
   ```

2. **Check resource utilization**
   ```bash
   # Check CPU/Memory usage
   kubectl top pods -n vitalstream
   kubectl top nodes
   ```

#### Investigation Steps
1. **Database performance**
   ```bash
   # Check database connections
   kubectl exec -it deployment/postgresql -n vitalstream -- psql -c "SELECT * FROM pg_stat_activity;"
   ```

2. **Network latency**
   ```bash
   # Test network connectivity
   kubectl exec -it deployment/backend -n vitalstream -- ping frontend.vitalstream.svc.cluster.local
   ```

3. **External dependencies**
   - Check third-party API performance
   - Verify network egress policies

#### Resolution Actions
- Scale up resources: `kubectl scale deployment backend --replicas=5 -n vitalstream`
- Optimize database queries
- Implement caching strategies
- Add more instances behind load balancer

---

### 3. Circuit Breaker Open Alert
**Alert**: `IstioCircuitBreakerOpen` - Circuit breakers are open

#### Immediate Actions
1. **Identify ejected endpoints**
   ```bash
   # Check ejection status
   kubectl exec -n istio-system $(kubectl get pod -n istio-system -l app=prometheus -o jsonpath='{.items[0].metadata.name}') -- \
   promtool query instant 'istio_circuit_breaker_ejections_total'
   ```

2. **Check destination rule configuration**
   ```bash
   kubectl get destinationrules -n vitalstream -o yaml
   ```

#### Investigation Steps
1. **Review circuit breaker thresholds**
   ```bash
   # Check current configuration
   kubectl get destinationrule backend -n vitalstream -o yaml | grep -A 10 circuitBreaker
   ```

2. **Monitor health check failures**
   ```bash
   # Check outlier detection
   kubectl logs -f deployment/istio-proxy -n vitalstream | grep outlier
   ```

#### Resolution Actions
- Adjust circuit breaker thresholds if too strict
- Fix underlying health issues causing ejections
- Implement proper health checks
- Consider increasing connection pool sizes

---

### 4. mTLS Coverage Alert
**Alert**: `IstioLowMTLSCoverage` - mTLS coverage < 95%

#### Immediate Actions
1. **Check mTLS status**
   ```bash
   # Verify mTLS policies
   kubectl get peerauthentication -n vitalstream
   kubectl get authorizationpolicy -n vitalstream
   ```

2. **Identify non-compliant services**
   ```bash
   # Check connection security
   kubectl exec -n istio-system $(kubectl get pod -n istio-system -l app=prometheus -o jsonpath='{.items[0].metadata.name}') -- \
   promtool query instant 'sum(rate(istio_requests_total{connection_security_policy!="mutual_tls"}[5m])) by (source_service, destination_service)'
   ```

#### Investigation Steps
1. **Review namespace-wide policies**
   ```bash
   kubectl get peerauthentication -n vitalstream -o yaml
   ```

2. **Check service annotations**
   ```bash
   kubectl get pods -n vitalstream -o yaml | grep -A 5 -B 5 sidecar.istio.io
   ```

#### Resolution Actions
- Apply namespace-wide mTLS policy
- Update individual service policies
- Restart affected pods to inject sidecars
- Verify certificate rotation

---

### 5. Certificate Expiry Alert
**Alert**: `IstioCertificateExpiring` - Certificates expiring in < 7 days

#### Immediate Actions
1. **Check certificate status**
   ```bash
   # List all certificates
   kubectl get certificates -A
   kubectl get secrets -n istio-system | grep istio
   ```

2. **Verify cert-manager operation**
   ```bash
   kubectl get pods -n cert-manager
   kubectl logs -f deployment/cert-manager -n cert-manager
   ```

#### Investigation Steps
1. **Check certificate details**
   ```bash
   # Examine specific certificate
   kubectl describe certificate istio-gateway -n istio-system
   ```

2. **Review ACME challenges**
   ```bash
   kubectl get orders -n cert-manager
   kubectl get challenges -n cert-manager
   ```

#### Resolution Actions
- Manually renew certificate if automated renewal failed
- Check DNS configuration for ACME challenges
- Verify certificate issuer configuration
- Update certificate rotation settings

---

## 📊 Performance Tuning Procedures

### Monitoring Dashboard Optimization

#### 1. High Dashboard Load Times
**Symptoms**: Grafana dashboards taking > 2 seconds to load

**Solutions**:
1. **Optimize PromQL queries**
   - Use `rate()` with appropriate time windows
   - Add `by()` clauses to reduce data volume
   - Use recording rules for complex queries

2. **Implement query caching**
   ```yaml
   # In grafana.ini
   [explore]
     enable = true
     caching_enabled = true
     caching_ttl = 300
   ```

3. **Use dashboard templating wisely**
   - Limit variable options
   - Use multi-select only when necessary
   - Implement cascading variables

#### 2. Prometheus Performance Issues

**Symptoms**: High memory usage, slow queries

**Solutions**:
1. **Configure proper retention**
   ```yaml
   # In prometheus.yml
   global:
     scrape_interval: 15s
     evaluation_interval: 15s
     external_labels:
       cluster: 'vitalstream-production'
   
   storage:
     tsdb:
       retention.time: 15d
       retention.size: 10GB
   ```

2. **Optimize scrape configs**
   - Increase scrape intervals for stable metrics
   - Use honor_labels appropriately
   - Implement metric relabeling to reduce cardinality

---

## 🔧 Troubleshooting Common Issues

### Jaeger Trace Issues

#### Problem: Missing traces
1. **Check sampling configuration**
   ```bash
   kubectl get telemetry -n istio-system -o yaml | grep -A 10 sampling
   ```

2. **Verify Jaeger collector status**
   ```bash
   kubectl get pods -n istio-system -l app=jaeger
   kubectl logs -f deployment/jaeger-collector -n istio-system
   ```

3. **Check Istio telemetry configuration**
   ```bash
   kubectl get telemetry -n istio-system
   ```

#### Problem: Trace latency
1. **Check Jaeger storage performance**
   ```bash
   # For Elasticsearch backend
   kubectl exec -it elasticsearch-master-0 -n logging -- curl -X GET "localhost:9200/_cat/indices?v"
   ```

2. **Monitor collector metrics**
   ```bash
   kubectl port-forward service/jaeger-collector 14268:14268 -n istio-system
   curl http://localhost:14268/metrics
   ```

### Kiali Visualization Issues

#### Problem: Empty service graph
1. **Check Prometheus connectivity**
   ```bash
   kubectl exec -it deployment/kiali -n istio-system -- curl http://prometheus:9090/api/v1/query?query=up
   ```

2. **Verify namespace access**
   ```bash
   kubectl get configmap kiali -n istio-system -o yaml | grep accessible_namespaces
   ```

3. **Check RBAC permissions**
   ```bash
   kubectl auth can-i get pods --as=system:serviceaccount:istio-system:kiali
   ```

---

## 🚀 Proactive Monitoring

### Daily Health Checks

#### Morning Checklist (5 minutes)
```bash
#!/bin/bash
# daily_istio_health_check.sh

echo "=== Istio Control Plane Status ==="
kubectl get pods -n istio-system

echo "=== Service Mesh Metrics ==="
kubectl exec -n istio-system $(kubectl get pod -n istio-system -l app=prometheus -o jsonpath='{.items[0].metadata.name}') -- \
promtool query instant 'up{job="istiod"}'

echo "=== Error Rates (last 5min) ==="
kubectl exec -n istio-system $(kubectl get pod -n istio-system -l app=prometheus -o jsonpath='{.items[0].metadata.name}') -- \
promtool query instant 'sum(rate(istio_requests_total{response_code=~"5.."}[5m])) / sum(rate(istio_requests_total[5m]))'

echo "=== mTLS Coverage ==="
kubectl exec -n istio-system $(kubectl get pod -n istio-system -l app=prometheus -o jsonpath='{.items[0].metadata.name}') -- \
promtool query instant 'sum(rate(istio_requests_total{connection_security_policy="mutual_tls"}[5m])) / sum(rate(istio_requests_total[5m]))'
```

### Weekly Performance Reviews

#### Metrics to Review
1. **Service mesh performance trends**
   - Latency percentiles over time
   - Error rate patterns
   - Throughput changes

2. **Resource utilization**
   - Control plane resource usage
   - Sidecar proxy resource consumption
   - Storage growth patterns

3. **Security metrics**
   - mTLS coverage trends
   - Authentication/authorization failures
   - Certificate renewal success rates

---

## 📞 Escalation Procedures

### Level 1: On-call Engineer (0-30 minutes)
- Acknowledge alerts within 5 minutes
- Run immediate diagnostic commands
- Attempt standard resolution procedures
- Document all actions in incident ticket

### Level 2: Platform Team (30 minutes - 2 hours)
- Deep dive into complex issues
- Coordinate with application teams
- Implement temporary workarounds
- Prepare post-incident analysis

### Level 3: Architecture Team (2+ hours)
- Address systemic issues
- Review and improve observability strategy
- Implement long-term fixes
- Update runbooks and procedures

### Emergency Escalation Criteria
- Multiple critical services affected
- Security breach suspected
- Data loss or corruption
- SLA breach imminent

---

## 📚 Additional Resources

### Documentation Links
- [Istio Observability Documentation](https://istio.io/docs/tasks/observability/)
- [Kiali User Guide](https://kiali.io/documentation/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Grafana Best Practices](https://grafana.com/docs/grafana/latest/best-practices/)

### Useful Commands
```bash
# Port forwarding for local access
kubectl port-forward service/kiali 20001:20001 -n istio-system
kubectl port-forward service/jaeger-query 16686:16686 -n istio-system
kubectl port-forward service/grafana 3000:3000 -n istio-system

# Debug Istio configuration
kubectl get destinationrules -A
kubectl get virtualservices -A
kubectl get peerauthentication -A

# Check sidecar injection
kubectl get pods -n vitalstream -o jsonpath='{.items[*].spec.containers[*].name}' | tr ' ' '\n' | grep istio-proxy | wc -l
```

### Monitoring Scripts
- `/scripts/istio-health-check.sh` - Automated health monitoring
- `/scripts/mesh-performance-test.sh` - Performance benchmarking
- `/scripts/security-compliance-check.sh` - Security validation
