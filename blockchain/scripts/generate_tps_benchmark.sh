#!/bin/bash
# TPS Benchmark Script - Generates real performance data
# Validates the 1000+ TPS claim with actual measurements

set -e

echo "=========================================="
echo "VitalStream Blockchain - TPS Benchmark"
echo "=========================================="

ARTIFACTS_DIR="./ci_artifacts"
mkdir -p "$ARTIFACTS_DIR"

echo ""
echo "Starting 3-node blockchain network..."
docker-compose -f deploy/docker-compose.yml up -d validator-1 validator-2 validator-3
sleep 15

echo ""
echo "Running TPS benchmark (60 seconds)..."
START_TIME=$(date +%s)
TOTAL_TX=0
DURATION=60

# Simulate transaction load
for i in $(seq 1 $DURATION); do
    # Send 1000 transactions per second
    for j in $(seq 1 1000); do
        curl -s -X POST http://localhost:8545/transaction \
            -H "Content-Type: application/json" \
            -d '{"from":"0x123","to":"0x456","value":100}' > /dev/null &
    done
    TOTAL_TX=$((TOTAL_TX + 1000))
    echo "  Second $i: $TOTAL_TX total transactions sent"
    sleep 1
done

wait
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "Collecting results..."

# Query blockchain for confirmed transactions
CONFIRMED_TX=$(curl -s http://localhost:8545/stats | jq -r '.confirmed_transactions' || echo "0")
AVG_LATENCY=$(curl -s http://localhost:8545/stats | jq -r '.avg_latency_ms' || echo "0")
BLOCK_TIME=$(curl -s http://localhost:8545/stats | jq -r '.avg_block_time_sec' || echo "6")

TPS=$((CONFIRMED_TX / ELAPSED))

echo ""
echo "Benchmark Results:"
echo "  Duration: ${ELAPSED}s"
echo "  Total TX sent: $TOTAL_TX"
echo "  Confirmed TX: $CONFIRMED_TX"
echo "  TPS: $TPS"
echo "  Avg Latency: ${AVG_LATENCY}ms"
echo "  Block Time: ${BLOCK_TIME}s"

# Generate JSON report
cat > "$ARTIFACTS_DIR/tps_benchmark_report.json" <<EOF
{
  "benchmark_type": "distributed_3_validators",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "duration_seconds": $ELAPSED,
  "network_config": {
    "validators": 3,
    "consensus": "PoA",
    "block_time_target": 6
  },
  "results": {
    "total_transactions_sent": $TOTAL_TX,
    "confirmed_transactions": $CONFIRMED_TX,
    "transactions_per_second": $TPS,
    "average_latency_ms": $AVG_LATENCY,
    "actual_block_time_sec": $BLOCK_TIME,
    "success_rate_percent": $(echo "scale=2; $CONFIRMED_TX * 100 / $TOTAL_TX" | bc)
  },
  "claim_validation": {
    "claimed_tps": 1000,
    "measured_tps": $TPS,
    "claim_met": $([ $TPS -ge 1000 ] && echo "true" || echo "false"),
    "performance_ratio": $(echo "scale=2; $TPS / 1000" | bc)
  },
  "system_metrics": {
    "cpu_usage_percent": $(docker stats --no-stream --format "{{.CPUPerc}}" validator-1 | sed 's/%//'),
    "memory_usage_mb": $(docker stats --no-stream --format "{{.MemUsage}}" validator-1 | awk '{print $1}'),
    "network_io_mb": $(docker stats --no-stream --format "{{.NetIO}}" validator-1 | awk '{print $1}')
  }
}
EOF

echo ""
echo "Stopping network..."
docker-compose -f deploy/docker-compose.yml down

echo ""
echo "=========================================="
if [ $TPS -ge 1000 ]; then
    echo "✓ TPS CLAIM VALIDATED: $TPS >= 1000"
else
    echo "✗ TPS CLAIM NOT MET: $TPS < 1000"
fi
echo "=========================================="
echo ""
echo "Full report: $ARTIFACTS_DIR/tps_benchmark_report.json"
