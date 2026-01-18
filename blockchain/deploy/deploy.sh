#!/bin/bash
# VitalStream Blockchain Deployment Script
# Production deployment automation

set -e

# Configuration
ENVIRONMENT=${1:-production}
CHAIN_SPEC="chain_spec.json"
DOCKER_COMPOSE="docker-compose.yml"
LOG_FILE="/var/log/vitalstream/deployment.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    command -v docker >/dev/null 2>&1 || error "Docker is not installed"
    command -v docker-compose >/dev/null 2>&1 || error "Docker Compose is not installed"
    
    if [ ! -f "$CHAIN_SPEC" ]; then
        error "Chain spec file not found: $CHAIN_SPEC"
    fi
    
    if [ ! -f "$DOCKER_COMPOSE" ]; then
        error "Docker Compose file not found: $DOCKER_COMPOSE"
    fi
    
    log "Prerequisites check passed"
}

# Generate validator keys
generate_validator_keys() {
    log "Generating validator keys..."
    
    for i in 1 2 3; do
        if [ ! -f "validator${i}-key.json" ]; then
            log "Generating key for validator ${i}..."
            docker run --rm vitalstream/blockchain-node:latest \
                key generate --scheme Ed25519 --output-type json \
                > "validator${i}-key.json"
        else
            warn "Validator ${i} key already exists, skipping"
        fi
    done
    
    log "Validator keys generated"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    docker build -t vitalstream/blockchain-node:latest -f Dockerfile ../../
    
    if [ $? -eq 0 ]; then
        log "Docker images built successfully"
    else
        error "Failed to build Docker images"
    fi
}

# Initialize chain
init_chain() {
    log "Initializing blockchain..."
    
    # Create genesis block
    docker run --rm \
        -v "$(pwd)/$CHAIN_SPEC:/chain_spec.json" \
        vitalstream/blockchain-node:latest \
        build-spec --chain /chain_spec.json --raw > chain_spec_raw.json
    
    log "Chain initialized"
}

# Deploy network
deploy_network() {
    log "Deploying blockchain network..."
    
    # Pull latest images
    docker-compose pull
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be healthy
    log "Waiting for services to be healthy..."
    sleep 30
    
    # Check validator status
    for i in 1 2 3; do
        port=$((9932 + i))
        if curl -s -f "http://localhost:$port/health" > /dev/null; then
            log "Validator $i is healthy"
        else
            warn "Validator $i health check failed"
        fi
    done
    
    log "Network deployed successfully"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check if all containers are running
    running=$(docker-compose ps -q | wc -l)
    expected=7  # 3 validators + 1 archive + prometheus + grafana + explorer
    
    if [ "$running" -eq "$expected" ]; then
        log "All containers are running ($running/$expected)"
    else
        warn "Some containers are not running ($running/$expected)"
    fi
    
    # Check consensus
    log "Checking consensus..."
    response=$(curl -s -X POST -H "Content-Type: application/json" \
        --data '{"jsonrpc":"2.0","method":"system_health","params":[],"id":1}' \
        http://localhost:9933)
    
    if echo "$response" | grep -q '"isSyncing":false'; then
        log "Node is synced"
    else
        warn "Node is still syncing"
    fi
    
    log "Deployment verification complete"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Configure Prometheus
    cat > prometheus.yml <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'vitalstream-validators'
    static_configs:
      - targets:
        - 'validator-1:9615'
        - 'validator-2:9615'
        - 'validator-3:9615'
        - 'archive-node:9615'
EOF
    
    log "Monitoring configured"
}

# Backup configuration
backup_config() {
    log "Backing up configuration..."
    
    backup_dir="backups/$(date +'%Y%m%d_%H%M%S')"
    mkdir -p "$backup_dir"
    
    cp "$CHAIN_SPEC" "$backup_dir/"
    cp validator*.json "$backup_dir/" 2>/dev/null || true
    cp "$DOCKER_COMPOSE" "$backup_dir/"
    
    log "Configuration backed up to $backup_dir"
}

# Main deployment flow
main() {
    log "Starting VitalStream Blockchain deployment ($ENVIRONMENT)"
    
    check_prerequisites
    backup_config
    generate_validator_keys
    build_images
    init_chain
    setup_monitoring
    deploy_network
    verify_deployment
    
    log "Deployment completed successfully!"
    log ""
    log "Access points:"
    log "  - RPC: http://localhost:9933"
    log "  - WebSocket: ws://localhost:9944"
    log "  - Prometheus: http://localhost:9090"
    log "  - Grafana: http://localhost:3000 (admin/admin)"
    log "  - Block Explorer: http://localhost:8080"
    log ""
    log "To view logs: docker-compose logs -f"
    log "To stop: docker-compose down"
}

# Run main function
main
