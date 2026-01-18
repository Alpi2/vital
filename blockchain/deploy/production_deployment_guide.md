# VitalStream Blockchain - Production Deployment Guide

## Critical Production Requirements - FIXED

This guide addresses all production blockers identified in the code review.

---

## 1. HSM/KMS Integration ✅ FIXED

### Issue
- **Before:** Keys stored in-memory only (key_management.rs:47)
- **After:** Full HSM/KMS support with 4 providers

### Implementation

```rust
// Production initialization
let hsm_config = HsmConfig::AwsKms {
    region: "us-east-1".to_string(),
    key_id: "arn:aws:kms:us-east-1:123456789:key/...".to_string(),
};

let key_manager = KeyManager::new_with_hsm(
    hsm_config,
    PathBuf::from("/var/lib/blockchain/revoked_tokens.json")
)?;
```

### Supported Providers
1. **AWS KMS** - Recommended for AWS deployments
2. **Azure Key Vault** - For Azure environments
3. **HashiCorp Vault** - For on-premise/hybrid
4. **PKCS#11 HSM** - For hardware HSM devices

### Configuration

```yaml
# Environment variables
HSM_PROVIDER=aws_kms
AWS_KMS_KEY_ID=arn:aws:kms:...
AWS_REGION=us-east-1
REVOCATION_DB_PATH=/var/lib/blockchain/revoked_tokens.json
```

---

## 2. TLS Network Security ✅ FIXED

### Issue
- **Before:** Plaintext TCP (peer_network.rs uses TcpStream)
- **After:** TLS 1.3 with mutual authentication

### Implementation

```rust
// Production initialization
let tls_config = TlsConfig {
    cert_path: "/etc/blockchain/certs/node.crt".to_string(),
    key_path: "/etc/blockchain/certs/node.key".to_string(),
    ca_cert_path: Some("/etc/blockchain/certs/ca.crt".to_string()),
};

let network = P2PNetwork::new_with_tls(
    node_id,
    listen_address,
    max_peers,
    tls_config
)?;
```

### Certificate Generation

```bash
# Generate CA
openssl req -x509 -newkey rsa:4096 -days 3650 \
    -keyout ca.key -out ca.crt -nodes \
    -subj "/CN=VitalStream Blockchain CA"

# Generate node certificate
openssl req -newkey rsa:4096 -nodes \
    -keyout node.key -out node.csr \
    -subj "/CN=validator-1.blockchain.local"

openssl x509 -req -in node.csr -CA ca.crt -CAkey ca.key \
    -CAcreateserial -out node.crt -days 365
```

### Features
- TLS 1.3 encryption
- Mutual authentication (mTLS)
- Certificate validation
- AES-256-GCM message encryption
- Perfect forward secrecy

---

## 3. Kubernetes Secrets Management ✅ FIXED

### Issue
- **Before:** Placeholder values in validator-deployment.yaml
- **After:** SealedSecrets + External Secrets Operator

### Option 1: Sealed Secrets (Recommended)

```bash
# Install Sealed Secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml

# Create secret
kubectl create secret generic validator-keys \
    --from-file=validator1-key.json \
    --from-file=validator2-key.json \
    --from-file=validator3-key.json \
    --dry-run=client -o yaml > secret.yaml

# Seal the secret
kubeseal --format=yaml < secret.yaml > sealed-secret.yaml

# Apply sealed secret
kubectl apply -f sealed-secret.yaml
```

### Option 2: External Secrets Operator (AWS)

```bash
# Install External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets

# Store secrets in AWS Secrets Manager
aws secretsmanager create-secret \
    --name vitalstream/blockchain/validator1 \
    --secret-string file://validator1-key.json

# Apply ExternalSecret (already in validator-deployment.yaml)
kubectl apply -f deploy/kubernetes/validator-deployment.yaml
```

### Option 3: HashiCorp Vault

```bash
# Store in Vault
vault kv put secret/blockchain/validator1 key=@validator1-key.json

# Configure Vault integration
kubectl apply -f deploy/kubernetes/vault-secretstore.yaml
```

---

## 4. CI/CD Artifacts Generation ✅ FIXED

### Issue
- **Before:** No proof of test results, benchmarks, or security audits
- **After:** Complete CI/CD pipeline with artifact generation

### Local Execution

```bash
# Run full CI/CD locally
chmod +x scripts/run_ci_locally.sh
./scripts/run_ci_locally.sh

# Artifacts generated in ./ci_artifacts/:
# - unit_test_results.txt
# - tarpaulin-report.json (coverage)
# - security_audit.json
# - benchmark_results.txt
# - hardhat_test_results.json
# - hardhat_coverage.json
# - integration_test_results.txt
```

### TPS Benchmark

```bash
# Run TPS benchmark
chmod +x scripts/generate_tps_benchmark.sh
./scripts/generate_tps_benchmark.sh

# Generates: ci_artifacts/tps_benchmark_report.json
```

### GitHub Actions

The pipeline automatically runs on push and generates artifacts:

```yaml
# .github/workflows/blockchain_ci.yml
- name: Upload artifacts
  uses: actions/upload-artifact@v3
  with:
    name: ci-artifacts
    path: ci_artifacts/
```

---

## 5. Production Deployment Checklist

### Pre-Deployment

- [ ] HSM/KMS configured and tested
- [ ] TLS certificates generated and distributed
- [ ] Kubernetes secrets encrypted (SealedSecrets or External Secrets)
- [ ] CI/CD pipeline executed successfully
- [ ] TPS benchmark meets requirements (≥1000 TPS)
- [ ] Security audit shows 0 critical/high issues
- [ ] Test coverage ≥90%

### Deployment

```bash
# 1. Create namespace
kubectl create namespace vitalstream-blockchain

# 2. Deploy secrets (using SealedSecrets)
kubectl apply -f deploy/kubernetes/sealed-secrets.yaml

# 3. Deploy validators
kubectl apply -f deploy/kubernetes/validator-deployment.yaml

# 4. Verify deployment
kubectl get pods -n vitalstream-blockchain
kubectl logs -n vitalstream-blockchain validator-1-0

# 5. Check consensus
kubectl exec -n vitalstream-blockchain validator-1-0 -- \
    curl http://localhost:8545/consensus/status
```

### Post-Deployment

- [ ] Monitor validator health
- [ ] Verify consensus participation
- [ ] Check TLS connections
- [ ] Monitor HSM/KMS usage
- [ ] Review audit logs
- [ ] Set up alerts (Prometheus/Grafana)

---

## 6. Monitoring & Observability

### Prometheus Metrics

```yaml
# Key metrics to monitor
- blockchain_tps
- blockchain_block_time_seconds
- blockchain_consensus_latency_ms
- blockchain_peer_count
- blockchain_hsm_operations_total
- blockchain_tls_connections_active
```

### Grafana Dashboards

```bash
# Import dashboard
kubectl apply -f deploy/kubernetes/grafana-dashboard.yaml
```

### Alerts

```yaml
# Critical alerts
- TPS < 1000 for 5 minutes
- Consensus failure
- HSM connection lost
- TLS certificate expiring (< 30 days)
- Validator offline
```

---

## 7. Disaster Recovery

### Backup

```bash
# Automated backup (CronJob)
kubectl apply -f deploy/kubernetes/backup-cronjob.yaml

# Manual backup
kubectl exec -n vitalstream-blockchain validator-1-0 -- \
    /usr/local/bin/blockchain-backup \
    --output /backups/blockchain-$(date +%Y%m%d).tar.gz
```

### Restore

```bash
# Restore from backup
kubectl exec -n vitalstream-blockchain validator-1-0 -- \
    /usr/local/bin/blockchain-restore \
    --input /backups/blockchain-20260104.tar.gz
```

---

## 8. Security Hardening

### Network Policies

```yaml
# Apply network policies
kubectl apply -f deploy/kubernetes/network-policies.yaml
```

### Pod Security

```yaml
# Enable Pod Security Standards
apiVersion: v1
kind: Namespace
metadata:
  name: vitalstream-blockchain
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

### RBAC

```bash
# Apply RBAC policies
kubectl apply -f deploy/kubernetes/rbac.yaml
```

---

## 9. Performance Tuning

### Validator Resources

```yaml
resources:
  requests:
    cpu: "4"
    memory: "8Gi"
  limits:
    cpu: "8"
    memory: "16Gi"
```

### Database Optimization

```bash
# RocksDB tuning
export ROCKSDB_MAX_OPEN_FILES=10000
export ROCKSDB_BLOCK_CACHE_SIZE=2147483648  # 2GB
export ROCKSDB_WRITE_BUFFER_SIZE=134217728   # 128MB
```

---

## 10. Compliance

### HIPAA
- ✅ Encryption at rest (RocksDB + AES-256-GCM)
- ✅ Encryption in transit (TLS 1.3)
- ✅ Audit logging (all transactions)
- ✅ Access control (HSM + RBAC)

### GDPR
- ✅ Data portability (export API)
- ✅ Right to erasure (anonymization)
- ✅ Data minimization (k-anonymity)
- ✅ Privacy by design (differential privacy)

---

## Support

For production deployment support:
- Email: blockchain-ops@vitalstream.com
- Slack: #blockchain-production
- On-call: +1-555-BLOCKCHAIN
