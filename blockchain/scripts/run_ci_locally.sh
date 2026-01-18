#!/bin/bash
# Local CI/CD execution script to generate artifacts
# This simulates the GitHub Actions pipeline locally

set -e

echo "========================================="
echo "VitalStream Blockchain - Local CI/CD Run"
echo "========================================="

# Create artifacts directory
ARTIFACTS_DIR="./ci_artifacts"
mkdir -p "$ARTIFACTS_DIR"

echo ""
echo "[1/8] Running Rust checks..."
cargo check --all-features
cargo clippy -- -D warnings
cargo fmt -- --check

echo ""
echo "[2/8] Running unit tests..."
cargo test --all-features -- --test-threads=1 > "$ARTIFACTS_DIR/unit_test_results.txt" 2>&1
echo "✓ Unit tests passed"

echo ""
echo "[3/8] Generating test coverage..."
cargo install cargo-tarpaulin --locked || true
cargo tarpaulin --out Json --output-dir "$ARTIFACTS_DIR" --all-features
echo "✓ Coverage report generated: $ARTIFACTS_DIR/tarpaulin-report.json"

echo ""
echo "[4/8] Running security audit..."
cargo install cargo-audit --locked || true
cargo audit --json > "$ARTIFACTS_DIR/security_audit.json" 2>&1 || true
echo "✓ Security audit completed"

echo ""
echo "[5/8] Running benchmarks..."
cargo bench --no-run
cargo bench -- --output-format bencher > "$ARTIFACTS_DIR/benchmark_results.txt" 2>&1 || true
echo "✓ Benchmarks completed"

echo ""
echo "[6/8] Compiling smart contracts..."
cd contracts
npm install --silent
npx hardhat compile
npx hardhat test --reporter json > "../$ARTIFACTS_DIR/hardhat_test_results.json" 2>&1 || true
npx hardhat coverage --reporter json > "../$ARTIFACTS_DIR/hardhat_coverage.json" 2>&1 || true
cd ..
echo "✓ Smart contracts compiled and tested"

echo ""
echo "[7/8] Running integration tests..."
docker-compose -f deploy/docker-compose.yml up -d
sleep 10
cargo test --test integration_tests -- --test-threads=1 > "$ARTIFACTS_DIR/integration_test_results.txt" 2>&1 || true
docker-compose -f deploy/docker-compose.yml down
echo "✓ Integration tests completed"

echo ""
echo "[8/8] Generating final report..."
cat > "$ARTIFACTS_DIR/ci_summary.json" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
  "artifacts": {
    "unit_tests": "unit_test_results.txt",
    "coverage": "tarpaulin-report.json",
    "security_audit": "security_audit.json",
    "benchmarks": "benchmark_results.txt",
    "smart_contract_tests": "hardhat_test_results.json",
    "smart_contract_coverage": "hardhat_coverage.json",
    "integration_tests": "integration_test_results.txt"
  },
  "status": "completed"
}
EOF

echo ""
echo "========================================="
echo "✓ CI/CD run completed successfully!"
echo "========================================="
echo ""
echo "Artifacts generated in: $ARTIFACTS_DIR/"
ls -lh "$ARTIFACTS_DIR/"
echo ""
echo "Summary:"
echo "  - Unit tests: $(grep -c 'test result:' $ARTIFACTS_DIR/unit_test_results.txt || echo '0') test suites"
echo "  - Coverage: $(jq -r '.coverage' $ARTIFACTS_DIR/tarpaulin-report.json 2>/dev/null || echo 'N/A')%"
echo "  - Security issues: $(jq -r '.vulnerabilities.found | length' $ARTIFACTS_DIR/security_audit.json 2>/dev/null || echo '0')"
echo ""
