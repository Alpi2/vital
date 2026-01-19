# Runbook: ML Inference Errors

## Severity
**WARNING / CRITICAL**

## Description
ML inference service is experiencing errors that affect model predictions and clinical decision support. This can impact patient monitoring accuracy and anomaly detection capabilities.

## Impact
- **Clinical Decision Support**: Reduced accuracy of ML predictions
- **Patient Safety**: Potential missed or false anomaly detections
- **Provider Experience**: Unreliable ML insights in dashboard
- **System Performance**: Increased error rates and potential service degradation

## Diagnosis

### 1. Check Grafana Dashboard
- **Dashboard**: [ML Performance Dashboard](https://grafana.vitalstream.com/d/vitalstream-ml-performance)
- **Key Metrics to Check**:
  - ML Inference Rate (should be stable)
  - ML Inference Latency (p95 should be <200ms)
  - Model Accuracy (should be >90%)
  - Error Rate by Model

### 2. Check Prometheus Metrics
```bash
# Check ML inference error rate
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(ml_inference_total{status=\"error\",environment=\"production\"}[5m])) by (model)"

# Check model accuracy
curl -s "http://prometheus:9090/api/v1/query?query=ml_model_accuracy{environment=\"production\"}"

# Check inference latency
curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(ml_inference_latency_seconds_bucket{environment=\"production\"}[5m]))"
```

### 3. Check Service Health
```bash
# Check ML inference service
curl -f http://ml-inference:8002/health || echo "ML inference service down"

# Check detailed health
curl -f http://ml-inference:8002/health/detailed || echo "Detailed health check failed"

# Check model loading status
curl -f http://ml-inference:8002/models/status || echo "Model status check failed"
```

### 4. Check Model Files
```bash
# Check if model files exist
kubectl exec -it deployment/ml-inference -n vitalstream -- ls -la /app/models/

# Check model file integrity
kubectl exec -it deployment/ml-inference -n vitalstream -- python -c "
import torch
try:
    model = torch.load('/app/models/arrhythmia_classifier.pt')
    print('Model loaded successfully')
    print(f'Model state dict keys: {list(model.state_dict().keys())[:5]}')
except Exception as e:
    print(f'Model loading failed: {e}')
"
```

### 5. Check GPU Resources (if applicable)
```bash
# Check GPU utilization
kubectl exec -it deployment/ml-inference -n vitalstream -- nvidia-smi

# Check GPU memory usage
kubectl exec -it deployment/ml-inference -n vitalstream -- nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Resolution Steps

### Step 1: Immediate Triage (First 5 minutes)
1. **Verify Alert Accuracy**
   ```bash
   # Check current error rate
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(ml_inference_total{status=\"error\",environment=\"production\"}[2m]))"
   ```

2. **Identify Affected Models**
   ```bash
   # Check which models are failing
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(ml_inference_total{status=\"error\",environment=\"production\"}[2m])) by (model)"
   ```

3. **Check Service Status**
   ```bash
   # Check if service is running
   kubectl get pods -l app=ml-inference -n vitalstream
   
   # Check pod logs
   kubectl logs -f -l app=ml-inference -n vitalstream --tail=50
   ```

### Step 2: Service Recovery (5-15 minutes)
1. **Restart ML Inference Service**
   ```bash
   # Restart deployment
   kubectl rollout restart deployment/ml-inference -n vitalstream
   
   # Check rollout status
   kubectl rollout status deployment/ml-inference -n vitalstream
   ```

2. **Check Model Loading**
   ```bash
   # Force model reload
   kubectl exec -it deployment/ml-inference -n vitalstream -- curl -X POST http://localhost:8002/models/reload
   
   # Check model status
   kubectl exec -it deployment/ml-inference -n vitalstream -- curl -s http://localhost:8002/models/status
   ```

3. **Clear Model Cache**
   ```bash
   # Clear model cache if corrupted
   kubectl exec -it deployment/ml-inference -n vitalstream -- rm -rf /tmp/model_cache/*
   
   # Restart service to reload cache
   kubectl rollout restart deployment/ml-inference -n vitalstream
   ```

### Step 3: Model-Specific Issues (15-30 minutes)
1. **Check Model Version**
   ```bash
   # Check current model version
   kubectl exec -it deployment/ml-inference -n vitalstream -- python -c "
   import torch
   model = torch.load('/app/models/arrhythmia_classifier.pt')
   print(f'Model version: {model.get(\"version\", \"unknown\")}')
   "
   ```

2. **Rollback Model Version**
   ```bash
   # Check available model versions
   kubectl exec -it deployment/ml-inference -n vitalstream -- ls -la /app/models/versions/
   
   # Rollback to previous version
   kubectl exec -it deployment/ml-inference -n vitalstream -- cp /app/models/versions/arrhythmia_classifier_v1.pt /app/models/arrhythmia_classifier.pt
   
   # Restart service
   kubectl rollout restart deployment/ml-inference -n vitalstream
   ```

3. **Re-train Model (if necessary)**
   ```bash
   # Trigger model retraining
   kubectl exec -it deployment/ml-inference -n vitalstream -- python -c "
   from ml_training.src.training.trainer import ModelTrainer
   trainer = ModelTrainer()
   trainer.retrain_model('arrhythmia_classifier')
   "
   ```

### Step 4: Resource Issues (30+ minutes)
1. **Scale Up Resources**
   ```bash
   # Increase replica count
   kubectl scale deployment/ml-inference --replicas=3 -n vitalstream
   
   # Increase resource limits
   kubectl patch deployment ml-inference -n vitalstream -p '{
     "spec": {
       "template": {
         "spec": {
           "containers": [{
             "name": "ml-inference",
             "resources": {
               "limits": {
                 "cpu": "4000m",
                 "memory": "8Gi",
                 "nvidia.com/gpu": "1"
               },
               "requests": {
                 "cpu": "2000m",
                 "memory": "4Gi",
                 "nvidia.com/gpu": "1"
               }
             }
           }]
         }
       }
     }
   }'
   ```

2. **Check GPU Availability**
   ```bash
   # Check GPU nodes
   kubectl get nodes -l accelerator=nvidia-tesla-v100
   
   # Check GPU resources
   kubectl describe nodes | grep -A 10 nvidia.com/gpu
   ```

3. **Implement Circuit Breaker**
   ```bash
   # Check if circuit breaker is configured
   kubectl get configmap ml-inference-config -n vitalstream -o yaml | grep circuit_breaker
   
   # Configure circuit breaker if not present
   kubectl patch configmap ml-inference-config -n vitalstream -p '{
     "data": {
       "circuit_breaker.yaml": "failure_threshold: 5\\nrecovery_timeout: 30\\nhalf_open_max_calls: 3"
     }
   }'
   ```

## Prevention

### 1. Model Monitoring
- Implement real-time model accuracy monitoring
- Set up model performance regression alerts
- Add model drift detection
- Implement automated model retraining

### 2. Resource Management
- Implement auto-scaling for ML inference
- Add GPU resource monitoring
- Implement model caching strategies
- Add resource usage alerts

### 3. Model Versioning
- Implement proper model versioning
- Add model A/B testing
- Implement gradual model rollouts
- Add model rollback procedures

### 4. Error Handling
- Implement graceful error degradation
- Add fallback model mechanisms
- Implement error categorization
- Add error recovery procedures

## Escalation

### Level 1 Escalation (After 15 minutes)
- **Contact**: ML Team
- **Slack**: #ml-team
- **Email**: ml-team@vitalstream.com
- **Phone**: +1-555-ML-TEAM

### Level 2 Escalation (After 30 minutes)
- **Contact**: ML Engineering Manager
- **Slack**: @ml-engineering-manager
- **Email**: ml-eng-manager@vitalstream.com
- **Phone**: +1-555-ML-ENG

### Level 3 Escalation (After 60 minutes)
- **Contact**: VP of AI/ML
- **Slack**: @vp-ai-ml
- **Email**: vp-ai-ml@vitalstream.com
- **Phone**: +1-555-VP-AIML

## Post-Incident Actions

1. **Model Analysis**
   - Analyze model performance degradation
   - Check training data quality
   - Evaluate model architecture
   - Document root causes

2. **Model Improvement**
   - Retrain models with new data
   - Optimize model architecture
   - Improve data preprocessing
   - Update model monitoring

3. **Process Improvement**
   - Update model deployment procedures
   - Improve model monitoring
   - Add automated testing
   - Update documentation

4. **Team Debrief**
   - Schedule post-mortem meeting
   - Discuss model performance issues
   - Identify improvement opportunities
   - Update best practices

## Related Documentation
- [ML Inference Architecture](https://docs.vitalstream.com/ml-inference)
- [Model Training Guide](https://docs.vitalstream.com/model-training)
- [GPU Optimization](https://docs.vitalstream.com/gpu-optimization)
- [Model Monitoring](https://docs.vitalstream.com/model-monitoring)

## Tools and Commands Reference

### Quick Commands
```bash
# Check ML inference service health
curl -f http://ml-inference:8002/health

# Check model status
curl -s http://ml-inference:8002/models/status

# Check error rate by model
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(ml_inference_total{status=\"error\",environment=\"production\"}[5m])) by (model)"

# Restart ML inference service
kubectl rollout restart deployment/ml-inference -n vitalstream

# Check GPU utilization
kubectl exec -it deployment/ml-inference -n vitalstream -- nvidia-smi

# Scale ML inference service
kubectl scale deployment/ml-inference --replicas=3 -n vitalstream
```

### Monitoring Queries
```promql
# ML inference error rate by model
sum(rate(ml_inference_total{status="error",environment="production"}[5m])) by (model)

# Model accuracy
ml_model_accuracy{environment="production"}

# Inference latency by model
histogram_quantile(0.95, rate(ml_inference_latency_seconds_bucket{environment="production"}[5m])) by (model)

# GPU utilization
rate(container_gpu_usage_seconds_total{container="ml-inference",environment="production"}[5m]) * 100

# Model loading time
histogram_quantile(0.95, rate(ml_model_loading_duration_seconds_bucket{environment="production"}[5m])) by (model)
```

---

**Last Updated**: 2026-01-16
**Version**: 1.0
**Maintainer**: ML Team
**Review Date**: 2026-04-16
