#!/bin/bash

# VitalStream Argo Rollouts Installation Script
# Enterprise-grade blue-green deployment with GitOps integration

set -euo pipefail

# Configuration
ARGO_ROLLOUTS_VERSION="v1.7.0"
ARGO_ROLLOUTS_NAMESPACE="argo-rollouts"
KUBECTL_VERSION="v1.28.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check cluster version
    K8S_VERSION=$(kubectl version --short | grep 'Server Version' | cut -d'v' -f2)
    log_info "Kubernetes version: $K8S_VERSION"
    
    if [[ "$(printf '%s\n' "1.24.0" "$K8S_VERSION" | sort -V | head -n1)" != "1.24.0" ]]; then
        log_error "Kubernetes version must be 1.24.0 or higher"
        exit 1
    fi
    
    # Check for ArgoCD
    if ! kubectl get namespace argocd &> /dev/null; then
        log_warning "ArgoCD not found. Please install ArgoCD first."
        log_info "Install ArgoCD with: ./install-argocd.sh"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating Argo Rollouts namespace..."
    
    kubectl create namespace argo-rollouts --dry-run=client -o yaml | kubectl apply -f -
    
    # Add labels for security and monitoring
    kubectl label namespace argo-rollouts \
        app.kubernetes.io/name=argo-rollouts \
        app.kubernetes.io/component=gitops \
        security.kubernetes.io/level=high \
        --overwrite
    
    log_success "Namespace created successfully"
}

# Install Argo Rollouts using Kustomize (recommended)
install_argo_rollouts() {
    log_info "Installing Argo Rollouts $ARGO_ROLLOUTS_VERSION..."
    
    # Download and apply Argo Rollouts manifests
    kubectl apply -n argo-rollouts \
        -f https://github.com/argoproj/argo-rollouts/releases/download/$ARGO_ROLLOUTS_VERSION/install.yaml
    
    # Wait for Rollouts controller to be ready
    log_info "Waiting for Rollouts controller to be ready..."
    kubectl wait --for=condition=available \
        deployment/argo-rollouts \
        --namespace argo-rollouts \
        --timeout=300s
    
    # Verify installation
    if kubectl api-resources | grep -q "rollouts.*argoproj.io/v1alpha1"; then
        log_success "Argo Rollouts installed successfully"
    else
        log_error "Argo Rollouts installation failed"
        exit 1
    fi
}

# Install Argo Rollouts CLI
install_argo_rollouts_cli() {
    log_info "Installing Argo Rollouts CLI..."
    
    # Determine architecture
    ARCH=$(uname -m | sed 's/x86_64/amd64/' | sed 's/arm64/arm64/')
    
    # Download CLI based on architecture and OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        CLI_URL="https://github.com/argoproj/argo-rollouts/releases/download/${ARGO_ROLLOUTS_VERSION}/kubectl-argo-rollouts-darwin-${ARCH}"
    else
        CLI_URL="https://github.com/argoproj/argo-rollouts/releases/download/${ARGO_ROLLOUTS_VERSION}/kubectl-argo-rollouts-linux-${ARCH}"
    fi
    
    # Download and install CLI
    curl -sSL -o kubectl-argo-rollouts "$CLI_URL"
    chmod +x kubectl-argo-rollouts
    
    # Install to system path
    if [[ -w "/usr/local/bin" ]]; then
        sudo mv kubectl-argo-rollouts /usr/local/bin/
    else
        mkdir -p ~/bin
        mv kubectl-argo-rollouts ~/bin/
        echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
        export PATH="$HOME/bin:$PATH"
    fi
    
    # Verify CLI installation
    if command -v kubectl-argo-rollouts &> /dev/null; then
        log_success "Argo Rollouts CLI installed successfully"
        kubectl-argo-rollouts version
    else
        log_error "Argo Rollouts CLI installation failed"
        exit 1
    fi
}

# Configure Argo Rollouts
configure_argo_rollouts() {
    log_info "Configuring Argo Rollouts..."
    
    # Create ConfigMap for Rollouts configuration
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: argo-rollouts-config
  namespace: argo-rollouts
  labels:
    app.kubernetes.io/name: argo-rollouts-config
    app.kubernetes.io/component: config
data:
  # Rollouts controller configuration
  controller: |
    # Metrics configuration
    metrics:
      enabled: true
      address: "0.0.0.0:8090"
      path: "/metrics"
    
    # Health check configuration
    health:
      enabled: true
      address: "0.0.0.0:8080"
      path: "/healthz"
    
    # Notification configuration
    notifications:
      enabled: true
      slack:
        token: "\$slack-token"
        channel: "#vitalstream-deployments"
        username: "Argo Rollouts"
        icon: ":shipit:"
    
    # Analysis configuration
    analysis:
      defaultTimeout: "10m"
      defaultRetryLimit: 3
      defaultBackoff:
        duration: "5s"
        factor: 2
        maxDuration: "3m"
    
    # Rollout configuration
    rollout:
      defaultReplicaCount: 3
      defaultRevisionHistoryLimit: 10
      defaultProgressDeadlineSeconds: 600
      defaultStrategy: "BlueGreen"
    
    # Service integration
    service:
      defaultTrafficRouting:
        enabled: true
        defaultStrategy: "BlueGreen"
        defaultTrafficSplit:
          active: 100
          preview: 0
    
    # Integration with ArgoCD
    argocd:
      enabled: true
      namespace: "argocd"
      syncOptions:
        - CreateNamespace=true
        - PrunePropagationPolicy=foreground
        - PruneLast=true
EOF

    log_success "Argo Rollouts configuration applied"
}

# Create RBAC for Rollouts
configure_rbac() {
    log_info "Configuring RBAC for Argo Rollouts..."
    
    # Create ServiceAccount and permissions
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: argo-rollouts
  namespace: argo-rollouts
  labels:
    app.kubernetes.io/name: argo-rollouts
    app.kubernetes.io/component: service-account
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: argo-rollouts
  labels:
    app.kubernetes.io/name: argo-rollouts
    app.kubernetes.io/component: rbac
rules:
# Core API permissions
- apiGroups: [""]
  resources: ["pods", "services", "endpoints", "events", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# Apps API permissions
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets", "daemonsets", "statefulsets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# Rollouts API permissions
- apiGroups: ["argoproj.io"]
  resources: ["rollouts", "rollouts/status", "rollouts/finalizers", "rollouts/scale"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# Analysis API permissions
- apiGroups: ["argoproj.io"]
  resources: ["analysistemplates", "analysistemplates/status", "analysistemplates/finalizers"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# Experiment API permissions
- apiGroups: ["argoproj.io"]
  resources: ["experiments", "experiments/status", "experiments/finalizers"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# Batch API permissions for analysis jobs
- apiGroups: ["batch"]
  resources: ["jobs", "cronjobs"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# Networking API permissions
- apiGroups: ["networking.k8s.io"]
  resources: ["ingresses", "networkpolicies"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# Autoscaling API permissions
- apiGroups: ["autoscaling"]
  resources: ["horizontalpodautoscalers"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: argo-rollouts
  labels:
    app.kubernetes.io/name: argo-rollouts
    app.kubernetes.io/component: rbac
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: argo-rollouts
subjects:
- kind: ServiceAccount
  name: argo-rollouts
  namespace: argo-rollouts
EOF

    log_success "RBAC configuration applied"
}

# Create Analysis Templates
create_analysis_templates() {
    log_info "Creating Analysis Templates..."
    
    # Smoke Tests Analysis Template
    kubectl apply -f - <<EOF
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: smoke-tests
  namespace: argo-rollouts
  labels:
    app.kubernetes.io/name: smoke-tests
    app.kubernetes.io/component: analysis-template
spec:
  args:
  - name: service-name
  - name: namespace
  - name: base-url
    value: "http://\$(args.service-name).\$(args.namespace).svc.cluster.local:8000"
  metrics:
  - name: smoke-test-health
    provider: job
    failureLimit: 1
    count: 1
    interval: 30s
    config:
      template:
        spec:
          containers:
          - name: smoke-test-health
            image: curlimages/curl:latest
            command:
            - sh
            - -c
            - |
              echo "🧪 Testing health endpoint..."
              if curl -f -s -m 10 "\$(args.base-url)/health" > /dev/null; then
                echo "✅ Health check passed"
                exit 0
              else
                echo "❌ Health check failed"
                exit 1
              fi
            env:
            - name: SERVICE_NAME
              value: "\$(args.service-name)"
            - name: NAMESPACE
              value: "\$(args.namespace)"
            - name: BASE_URL
              value: "\$(args.base-url)"
            resources:
              requests:
                cpu: 100m
                memory: 128Mi
              limits:
                cpu: 200m
                memory: 256Mi
            securityContext:
              allowPrivilegeEscalation: false
              readOnlyRootFilesystem: true
              runAsNonRoot: true
              runAsUser: 1000
              runAsGroup: 1000
          restartPolicy: Never
          backoffLimit: 3
          ttlSecondsAfterFinished: 300
  
  - name: smoke-test-auth
    provider: job
    failureLimit: 1
    count: 1
    interval: 30s
    config:
      template:
        spec:
          containers:
          - name: smoke-test-auth
            image: curlimages/curl:latest
            command:
            - sh
            - -c
            - |
              echo "🧪 Testing authentication..."
              TOKEN=\$(curl -s -X POST "\$(args.base-url)/api/v1/auth/login" \
                -H "Content-Type: application/json" \
                -d '{"username":"\$API_USERNAME","password":"\$API_PASSWORD"}' | jq -r '.access_token')
              
              if [ -n "\$TOKEN" ] && [ "\$TOKEN" != "null" ]; then
                echo "✅ Authentication test passed"
                exit 0
              else
                echo "❌ Authentication test failed"
                exit 1
              fi
            env:
            - name: SERVICE_NAME
              value: "\$(args.service-name)"
            - name: NAMESPACE
              value: "\$(args.namespace)"
            - name: BASE_URL
              value: "\$(args.base-url)"
            - name: API_USERNAME
              valueFrom:
                secretKeyRef:
                  name: vitalstream-secrets
                  key: api-username
            - name: API_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: vitalstream-secrets
                  key: api-password
            resources:
              requests:
                cpu: 100m
                memory: 128Mi
              limits:
                cpu: 200m
                memory: 256Mi
            securityContext:
              allowPrivilegeEscalation: false
              readOnlyRootFilesystem: true
              runAsNonRoot: true
              runAsUser: 1000
              runAsGroup: 1000
          restartPolicy: Never
          backoffLimit: 3
          ttlSecondsAfterFinished: 300
  
  - name: smoke-test-api
    provider: job
    failureLimit: 1
    count: 1
    interval: 30s
    config:
      template:
        spec:
          containers:
          - name: smoke-test-api
            image: curlimages/curl:latest
            command:
            - sh
            - -c
            - |
              echo "🧪 Testing API endpoints..."
              TOKEN=\$(curl -s -X POST "\$(args.base-url)/api/v1/auth/login" \
                -H "Content-Type: application/json" \
                -d '{"username":"\$API_USERNAME","password":"\$API_PASSWORD"}' | jq -r '.access_token')
              
              # Test ECG analysis endpoint
              if curl -f -s -X POST "\$(args.base-url)/api/v1/ecg/analyze" \
                -H "Authorization: Bearer \$TOKEN" \
                -H "Content-Type: application/json" \
                -d '{"patient_id":"test","samples":[0.1,0.2,0.3],"sampling_rate":360}' > /dev/null; then
                echo "✅ ECG analysis test passed"
                exit 0
              else
                echo "❌ ECG analysis test failed"
                exit 1
              fi
            env:
            - name: SERVICE_NAME
              value: "\$(args.service-name)"
            - name: NAMESPACE
              value: "\$(args.namespace)"
            - name: BASE_URL
              value: "\$(args.base-url)"
            - name: API_USERNAME
              valueFrom:
                secretKeyRef:
                  name: vitalstream-secrets
                  key: api-username
            - name: API_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: vitalstream-secrets
                  key: api-password
            resources:
              requests:
                cpu: 100m
                memory: 128Mi
              limits:
                cpu: 200m
                memory: 256Mi
            securityContext:
              allowPrivilegeEscalation: false
              readOnlyRootFilesystem: true
              runAsNonRoot: true
              runAsUser: 1000
              runAsGroup: 1000
          restartPolicy: Never
          backoffLimit: 3
          ttlSecondsAfterFinished: 300
  
  - name: smoke-test-database
    provider: job
    failureLimit: 1
    count: 1
    interval: 30s
    config:
      template:
        spec:
          containers:
          - name: smoke-test-database
            image: curlimages/curl:latest
            command:
            - sh
            - -c
            - |
              echo "🧪 Testing database connectivity..."
              TOKEN=\$(curl -s -X POST "\$(args.base-url)/api/v1/auth/login" \
                -H "Content-Type: application/json" \
                -d '{"username":"\$API_USERNAME","password":"\$API_PASSWORD"}' | jq -r '.access_token')
              
              # Test database endpoint
              if curl -f -s "\$(args.base-url)/api/v1/patients?limit=1" \
                -H "Authorization: Bearer \$TOKEN" > /dev/null; then
                echo "✅ Database connectivity test passed"
                exit 0
              else
                echo "❌ Database connectivity test failed"
                exit 1
              fi
            env:
            - name: SERVICE_NAME
              value: "\$(args.service-name)"
            - name: NAMESPACE
              value: "\$(args.namespace)"
            - name: BASE_URL
              value: "\$(args.base-url)"
            - name: API_USERNAME
              valueFrom:
                secretKeyRef:
                  name: vitalstream-secrets
                  key: api-username
            - name: API_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: vitalstream-secrets
                  key: api-password
            resources:
              requests:
                cpu: 100m
                memory: 128Mi
              limits:
                cpu: 200m
                memory: 256Mi
            securityContext:
              allowPrivilegeEscalation: false
              readOnlyRootFilesystem: true
              runAsNonRoot: true
              runAsUser: 1000
              runAsGroup: 1000
          restartPolicy: Never
          backoffLimit: 3
          ttlSecondsAfterFinished: 300
EOF

    # SLO Analysis Template
    kubectl apply -f - <<EOF
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: slo-analysis
  namespace: argo-rollouts
  labels:
    app.kubernetes.io/name: slo-analysis
    app.kubernetes.io/component: analysis-template
spec:
  args:
  - name: service-name
  - name: namespace
  - name: slo-error-rate
    value: "0.01"
  - name: slo-latency-p95
    value: "0.5"
  metrics:
  - name: error-rate
    provider: prometheus
    failureLimit: 1
    count: 1
    interval: 30s
    successCondition: result[0] < args.slo-error-rate
    config:
      queries:
      - name: error-rate
        query: |
          sum(rate(http_requests_total{service="\$(args.service-name)",status=~"5.."}[5m])) / 
          sum(rate(http_requests_total{service="\$(args.service-name)"}[5m]))
  
  - name: latency-p95
    provider: prometheus
    failureLimit: 1
    count: 1
    interval: 30s
    successCondition: result[0] < args.slo-latency-p95
    config:
      queries:
      - name: latency-p95
        query: |
          histogram_quantile(0.95, 
            sum(rate(http_request_duration_seconds_bucket{service="\$(args.service-name)"}[5m])) by (le))
EOF

    log_success "Analysis Templates created successfully"
}

# Create secrets for Rollouts
create_secrets() {
    log_info "Creating secrets for Argo Rollouts..."
    
    # Create Rollouts secrets
    kubectl create secret generic argo-rollouts-secrets \
        --namespace argo-rollouts \
        --from-literal=slack-token="\$SLACK_TOKEN" \
        --from-literal=prometheus-token="\$PROMETHEUS_TOKEN" \
        --from-literal=notification-webhook="\$NOTIFICATION_WEBHOOK" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Secrets created successfully"
}

# Verify installation
verify_installation() {
    log_info "Verifying Argo Rollouts installation..."
    
    # Wait for Rollouts controller to be ready
    kubectl wait --for=condition=available \
        deployment/argo-rollouts \
        --namespace argo-rollouts \
        --timeout=300s
    
    # Check API resources
    if kubectl api-resources | grep -q "rollouts.*argoproj.io/v1alpha1"; then
        log_success "Rollouts API is available"
    else
        log_error "Rollouts API is not available"
        exit 1
    fi
    
    # Check Analysis Templates
    if kubectl get analysistemplates -n argo-rollouts | grep -q "smoke-tests"; then
        log_success "Analysis Templates are available"
    else
        log_error "Analysis Templates are not available"
        exit 1
    fi
    
    # Test CLI
    if command -v kubectl-argo-rollouts &> /dev/null; then
        log_success "Argo Rollouts CLI is working"
        kubectl-argo-rollouts version
    else
        log_error "Argo Rollouts CLI is not working"
        exit 1
    fi
    
    log_success "Argo Rollouts installation verified successfully"
}

# Main installation function
main() {
    log_info "Starting VitalStream Argo Rollouts installation..."
    
    check_prerequisites
    create_namespace
    install_argo_rollouts
    install_argo_rollouts_cli
    configure_argo_rollouts
    configure_rbac
    create_analysis_templates
    create_secrets
    verify_installation
    
    log_success "Argo Rollouts installation completed successfully!"
    log_info "Next steps:"
    log_info "1. Convert Deployments to Rollouts in Helm charts"
    log_info "2. Configure ArgoCD to sync Rollouts"
    log_info "3. Test blue-green deployments"
    log_info "4. Set up monitoring and alerting"
}

# Run main function
main "$@"
