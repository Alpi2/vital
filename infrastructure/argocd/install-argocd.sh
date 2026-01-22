#!/bin/bash

# VitalStream ArgoCD Installation Script
# Enterprise-grade GitOps platform with security best practices

set -euo pipefail

# Configuration
ARGOCD_VERSION="v2.9.5"
ARGOCD_NAMESPACE="argocd"
ARGOCD_HELM_VERSION="2.9.5"
ARCH=$(uname -m | sed 's/x86_64/amd64/' | sed 's/arm64/arm64/')

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
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "Helm is not installed"
        exit 1
    fi
    
    # Check cluster version
    K8S_VERSION=$(kubectl version --short | grep 'Server Version' | cut -d' ' -f3 | cut -d'v' -f2)
    log_info "Kubernetes version: $K8S_VERSION"
    
    if [[ "$(printf '%s\n' "1.24.0" "$K8S_VERSION" | sort -V | head -n1)" != "1.24.0" ]]; then
        log_error "Kubernetes version must be 1.24.0 or higher"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating ArgoCD namespace..."
    
    kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
    
    # Add labels for security and monitoring
    kubectl label namespace argocd \
        app.kubernetes.io/name=argocd \
        app.kubernetes.io/component=gitops \
        security.kubernetes.io/level=high \
        --overwrite
    
    log_success "Namespace created successfully"
}

# Install ArgoCD using Helm (recommended for production)
install_argocd_helm() {
    log_info "Installing ArgoCD $ARGOCD_VERSION using Helm..."
    
    # Add ArgoCD Helm repository
    helm repo add argo https://argoproj.github.io/argo-helm
    helm repo update
    
    # Create values file for production
    cat > argocd-values.yaml <<EOF
global:
  image:
    tag: $ARGOCD_VERSION
  securityContext:
    runAsNonRoot: true
    runAsUser: 999
    fsGroup: 999

server:
  # Ingress configuration
  ingress:
    enabled: true
    ingressClassName: nginx
    annotations:
      cert-manager.io/cluster-issuer: letsencrypt-prod
      nginx.ingress.kubernetes.io/ssl-redirect: "true"
      nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
      nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
      nginx.ingress.kubernetes.io/configuration-snippet: |
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header X-Forwarded-Host \$host;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    hosts:
      - argocd.vitalstream.com
    tls:
      - secretName: argocd-server-tls
        hosts:
          - argocd.vitalstream.com
  
  # RBAC configuration
  rbacConfig:
    policy.csv: |
      p, role:admin, applications, *, */*, allow
      p, role:admin, clusters, *, *, allow
      p, role:readonly, applications, get, */*, allow
      p, role:readonly, clusters, get, *, allow
      g, vitalstream, role:admin
      g, developers, role:readonly
  
  # Service configuration
  service:
    type: ClusterIP
    port: 80
    targetPort: 8080
  
  # Resource limits
  resources:
    limits:
      cpu: 500m
      memory: 512Mi
    requests:
      cpu: 250m
      memory: 256Mi
  
  # Security settings
  insecure: false
  config:
    # Enable SSO with Dex
    dex.config: |
      connectors:
        - type: oidc
          id: oidc
          name: VitalStream SSO
          config:
            issuer: https://sso.vitalstream.com
            clientID: argocd
            clientSecret: \$oidc.clientSecret
            requestedScopes: ["openid", "profile", "email", "groups"]
    
    # Enable audit logging
    application.instanceLabelKey: argocd.argoproj.io/instance
    
    # Performance settings
    timeout.reconciliation: 180s
    timeout.hard.reconciliation: 600s

# Redis configuration
redis:
  enabled: true
  resources:
    limits:
      cpu: 200m
      memory: 256Mi
    requests:
      cpu: 100m
      memory: 128Mi

# Controller configuration
controller:
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi
    requests:
      cpu: 500m
      memory: 512Mi
  
  # Performance settings
  processors:
    operation: 10
    status: 20
  
  # Sync settings
  sync:
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m

# Repo server configuration
repoServer:
  resources:
    limits:
      cpu: 500m
      memory: 512Mi
    requests:
      cpu: 250m
      memory: 256Mi
  
  # Mount credentials
  volumes:
    - name: git-credentials
      secret:
        secretName: argocd-repo-creds
  volumeMounts:
    - name: git-credentials
      mountPath: /app/config/ssh
      readOnly: true

# ApplicationSet controller
applicationSet:
  enabled: true
  resources:
    limits:
      cpu: 500m
      memory: 512Mi
    requests:
      cpu: 250m
      memory: 256Mi

# Notifications controller
notifications:
  enabled: true
  resources:
    limits:
      cpu: 200m
      memory: 256Mi
    requests:
      cpu: 100m
      memory: 128Mi

# SSO configuration
configs:
  cm:
    # Enable RBAC
    accounts.vitalstream: login
    accounts.developers: login
    accounts.rbac: groups
  
  # SSO secret
  secret:
    createSecret: true
    argocdServerAdminPassword: "\$2a\$10\$abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
    oidc.clientSecret: "\$oidc.clientSecret"

# Monitoring
metrics:
  enabled: true
  serviceMonitor:
    enabled: true
    namespace: monitoring
    labels:
      release: prometheus
EOF

    # Install ArgoCD
    helm upgrade --install argocd argo/argo-cd \
        --namespace argocd \
        --values argocd-values.yaml \
        --wait \
        --timeout 10m
    
    log_success "ArgoCD installed successfully using Helm"
}

# Install ArgoCD CLI
install_argocd_cli() {
    log_info "Installing ArgoCD CLI..."
    
    # Download CLI based on architecture
    CLI_URL="https://github.com/argoproj/argo-cd/releases/download/${ARGOCD_VERSION}/argocd-linux-${ARCH}"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        CLI_URL="https://github.com/argoproj/argo-cd/releases/download/${ARGOCD_VERSION}/argocd-darwin-${ARCH}"
    fi
    
    curl -sSL -o argocd "$CLI_URL"
    chmod +x argocd
    
    # Install to system path
    if [[ -w "/usr/local/bin" ]]; then
        sudo mv argocd /usr/local/bin/
    else
        mkdir -p ~/bin
        mv argocd ~/bin/
        echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
        export PATH="$HOME/bin:$PATH"
    fi
    
    log_success "ArgoCD CLI installed successfully"
}

# Configure RBAC
configure_rbac() {
    log_info "Configuring RBAC..."
    
    # Create RBAC ConfigMap
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: argocd-rbac-cm
  namespace: argocd
  labels:
    app.kubernetes.io/name: argocd-rbac-cm
    app.kubernetes.io/component: rbac
data:
  policy.csv: |
    p, role:admin, applications, *, */*, allow
    p, role:admin, clusters, *, *, allow
    p, role:readonly, applications, get, */*, allow
    p, role:readonly, clusters, get, *, allow
    p, role:developer, applications, *, vitalstream-*, allow
    p, role:developer, clusters, get, *, allow
    p, role:ops, applications, sync, *, allow
    p, role:ops, applications, override, *, allow
    p, role:ops, applications, action, *, allow
    g, vitalstream-admins, role:admin
    g, vitalstream-developers, role:developer
    g, vitalstream-ops, role:ops
    g, vitalstream-viewers, role:readonly
  policy.default: role:readonly
EOF

    log_success "RBAC configured successfully"
}

# Create secrets
create_secrets() {
    log_info "Creating secrets..."
    
    # Create GitHub credentials secret
    kubectl create secret generic argocd-repo-creds \
        --namespace argocd \
        --from-literal=type=git \
        --from-literal=url=https://github.com/vitalstream/vitalstream.git \
        --from-literal=username=vitalstream-bot \
        --from-literal=password=\$GITHUB_TOKEN \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create notification secrets
    kubectl create secret generic argocd-notifications-secret \
        --namespace argocd \
        --from-literal=slack-token=\$SLACK_TOKEN \
        --from-literal=email-smtp-host=\$SMTP_HOST \
        --from-literal=email-smtp-port=\$SMTP_PORT \
        --from-literal=email-username=\$SMTP_USERNAME \
        --from-literal=email-password=\$SMTP_PASSWORD \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Secrets created successfully"
}

# Configure notifications
configure_notifications() {
    log_info "Configuring notifications..."
    
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: argocd-notifications-cm
  namespace: argocd
  labels:
    app.kubernetes.io/name: argocd-notifications-cm
    app.kubernetes.io/component: notifications
data:
  service.slack: |
    token: \$slack-token
    username: ArgoCD
    icon: :argo:
  trigger.on-deployed: |
    - when: app.status.operationState.phase in ['Succeeded']
      send: [app-deployed]
  trigger.on-health-degraded: |
    - when: app.status.health.status == 'Degraded'
      send: [app-health-degraded]
  trigger.on-sync-failed: |
    - when: app.status.operationState.phase in ['Error', 'Failed']
      send: [app-sync-failed]
  trigger.on-sync-running: |
    - when: app.status.operationState.phase in ['Running']
      send: [app-sync-running]
  template.app-deployed: |
    message: |
      ✅ Application {{.app.metadata.name}} deployed successfully!
      Environment: {{.app.spec.destination.namespace}}
      Revision: {{.app.status.sync.revision}}
      Commit: {{.app.status.operationState.syncResult.revision}}
    slack:
      attachments: |
        [{
          "title": "Deployment Success",
          "color": "good",
          "fields": [
            {
              "title": "Application",
              "value": "{{.app.metadata.name}}",
              "short": true
            },
            {
              "title": "Environment",
              "value": "{{.app.spec.destination.namespace}}",
              "short": true
            },
            {
              "title": "Revision",
              "value": "{{.app.status.sync.revision}}",
              "short": true
            }
          ]
        }]
  template.app-health-degraded: |
    message: |
      🚨 Application {{.app.metadata.name}} health is degraded!
      Environment: {{.app.spec.destination.namespace}}
      Status: {{.app.status.health.status}}
    slack:
      attachments: |
        [{
          "title": "Health Degraded",
          "color": "danger",
          "fields": [
            {
              "title": "Application",
              "value": "{{.app.metadata.name}}",
              "short": true
            },
            {
              "title": "Environment",
              "value": "{{.app.spec.destination.namespace}}",
              "short": true
            },
            {
              "title": "Status",
              "value": "{{.app.status.health.status}}",
              "short": true
            }
          ]
        }]
  template.app-sync-failed: |
    message: |
      ❌ Application {{.app.metadata.name}} sync failed!
      Environment: {{.app.spec.destination.namespace}}
      Error: {{.app.status.operationState.message}}
    slack:
      attachments: |
        [{
          "title": "Sync Failed",
          "color": "danger",
          "fields": [
            {
              "title": "Application",
              "value": "{{.app.metadata.name}}",
              "short": true
            },
            {
              "title": "Environment",
              "value": "{{.app.spec.destination.namespace}}",
              "short": true
            },
            {
              "title": "Error",
              "value": "{{.app.status.operationState.message}}",
              "short": false
            }
          ]
        }]
EOF

    log_success "Notifications configured successfully"
}

# Verify installation
verify_installation() {
    log_info "Verifying ArgoCD installation..."
    
    # Wait for ArgoCD to be ready
    kubectl wait --for=condition=available \
        deployment/argocd-server \
        --namespace argocd \
        --timeout=300s
    
    # Get initial admin password
    ADMIN_PASSWORD=$(kubectl -n argocd get secret argocd-initial-admin-secret \
        -o jsonpath="{.data.password}" | base64 -d)
    
    log_success "ArgoCD installation verified successfully"
    log_info "Initial admin password: $ADMIN_PASSWORD"
    log_info "ArgoCD URL: https://argocd.vitalstream.com"
    
    # Clean up temporary files
    rm -f argocd-values.yaml
}

# Main installation function
main() {
    log_info "Starting VitalStream ArgoCD installation..."
    
    check_prerequisites
    create_namespace
    install_argocd_helm
    install_argocd_cli
    configure_rbac
    create_secrets
    configure_notifications
    verify_installation
    
    log_success "ArgoCD installation completed successfully!"
    log_info "Next steps:"
    log_info "1. Access ArgoCD UI: https://argocd.vitalstream.com"
    log_info "2. Login with admin credentials"
    log_info "3. Configure Git repositories"
    log_info "4. Deploy applications using ApplicationSets"
}

# Run main function
main "$@"
