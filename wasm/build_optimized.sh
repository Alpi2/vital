#!/bin/bash
# Optimized WASM build script with caching

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SOURCE_DIR="src"
BUILD_DIR="build"
CACHE_DIR=".build-cache"
# Standardized output path to Angular assets (served at /assets/wasm/...)
OUTPUT_DIR="../frontend/src/assets/wasm/ecg_processor"
OPTIMIZATION_LEVEL="-O3"

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}🔨 VitalStream WASM Build (Optimized with Caching)${NC}"
echo "================================================="

# Function to calculate file hash
calculate_hash() {
    if command -v md5sum &> /dev/null; then
        find "$SOURCE_DIR" -type f \( -name "*.cpp" -o -name "*.h" \) -exec md5sum {} \; | sort | md5sum | cut -d' ' -f1
    elif command -v md5 &> /dev/null; then
        find "$SOURCE_DIR" -type f \( -name "*.cpp" -o -name "*.h" \) -exec md5 -q {} \; | sort | md5 -q
    else
        echo "none"
    fi
}

# Function to check if rebuild is needed
needs_rebuild() {
    local current_hash=$(calculate_hash)
    local cache_file="$CACHE_DIR/build.hash"
    
    if [ ! -f "$cache_file" ]; then
        echo -e "${YELLOW}📝 No cache found, full build required${NC}"
        return 0
    fi
    
    local cached_hash=$(cat "$cache_file")
    
    if [ "$current_hash" != "$cached_hash" ]; then
        echo -e "${YELLOW}🔄 Source files changed, rebuild required${NC}"
        echo "  Previous: $cached_hash"
        echo "  Current:  $current_hash"
        return 0
    fi
    
    if [ ! -f "$OUTPUT_DIR/ecg_analyzer.wasm" ]; then
        echo -e "${YELLOW}📦 Output files missing, rebuild required${NC}"
        return 0
    fi
    
    echo -e "${GREEN}✅ Cache hit! No rebuild needed${NC}"
    return 1
}

# Check if emscripten is available
if ! command -v emcc &> /dev/null; then
    echo -e "${RED}❌ Error: emcc not found${NC}"
    echo "Please install Emscripten SDK:"
    echo "  git clone https://github.com/emscripten-core/emsdk.git"
    echo "  cd emsdk"
    echo "  ./emsdk install latest"
    echo "  ./emsdk activate latest"
    echo "  source ./emsdk_env.sh"
    exit 1
fi

echo -e "${GREEN}✓ Emscripten found: $(emcc --version | head -n1)${NC}"

# Check if rebuild is needed
if ! needs_rebuild; then
    echo -e "${GREEN}🎉 Using cached build${NC}"
    exit 0
fi

echo -e "${YELLOW}🔨 Starting build...${NC}"

# Build timestamp
BUILD_START=$(date +%s)

# Compile C++ to WASM
echo -e "${YELLOW}📦 Compiling C++ to WebAssembly...${NC}"

emcc \
    "$SOURCE_DIR/ecg_analyzer.cpp" \
    "$SOURCE_DIR/signal_processing.cpp" \
    "$SOURCE_DIR/anomaly_detector.cpp" \
    $OPTIMIZATION_LEVEL \
    -s WASM=1 \
    -s EXPORTED_FUNCTIONS='["_analyze_ecg","_detect_anomalies","_calculate_heart_rate","_filter_signal","_malloc","_free"]' \
    -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue"]' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=16MB \
    -s MAXIMUM_MEMORY=256MB \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="createECGAnalyzer" \
    -s ENVIRONMENT='web,worker' \
    -s FILESYSTEM=0 \
    -s ASSERTIONS=0 \
    -s MALLOC=emmalloc \
    --no-entry \
    -o "$BUILD_DIR/ecg_analyzer.js"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Compilation successful${NC}"

# Optimize WASM file
if command -v wasm-opt &> /dev/null; then
    echo -e "${YELLOW}⚡ Optimizing WASM with wasm-opt...${NC}"
    wasm-opt -O3 -o "$BUILD_DIR/ecg_analyzer_opt.wasm" "$BUILD_DIR/ecg_analyzer.wasm"
    mv "$BUILD_DIR/ecg_analyzer_opt.wasm" "$BUILD_DIR/ecg_analyzer.wasm"
    echo -e "${GREEN}✓ Optimization complete${NC}"
else
    echo -e "${YELLOW}⚠ wasm-opt not found, skipping optimization${NC}"
    echo "  Install binaryen for better optimization:"
    echo "  brew install binaryen  # macOS"
    echo "  sudo apt install binaryen  # Ubuntu"
fi

# Copy to standardized Angular assets directory
echo -e "${YELLOW}📋 Copying files to Angular assets directory: $OUTPUT_DIR${NC}"
mkdir -p "$OUTPUT_DIR"
cp "$BUILD_DIR/ecg_analyzer.js" "$OUTPUT_DIR/"
cp "$BUILD_DIR/ecg_analyzer.wasm" "$OUTPUT_DIR/"

# Generate TypeScript definitions alongside outputs
echo -e "${YELLOW}📝 Generating TypeScript definitions...${NC}"
cat > "$OUTPUT_DIR/ecg_analyzer.d.ts" << 'EOF'
/**
 * ECG Analyzer WebAssembly Module
 * Auto-generated TypeScript definitions
 */

export interface ECGAnalyzerModule extends EmscriptenModule {
  ccall: typeof ccall;
  cwrap: typeof cwrap;
  getValue: typeof getValue;
  setValue: typeof setValue;
  _malloc: (size: number) => number;
  _free: (ptr: number) => void;
}

export interface ECGAnalysisResult {
  heartRate: number;
  rrIntervals: number[];
  qrsComplexes: number[];
  anomalies: AnomalyDetection[];
}

export interface AnomalyDetection {
  type: 'bradycardia' | 'tachycardia' | 'arrhythmia' | 'st_elevation';
  timestamp: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
}

export default function createECGAnalyzer(): Promise<ECGAnalyzerModule>;
EOF

# Calculate file sizes
JS_SIZE=$(du -h "$OUTPUT_DIR/ecg_analyzer.js" | cut -f1)
WASM_SIZE=$(du -h "$OUTPUT_DIR/ecg_analyzer.wasm" | cut -f1)

# Save build hash
calculate_hash > "$CACHE_DIR/build.hash"

# Save build metadata
cat > "$CACHE_DIR/build.json" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "hash": "$(cat $CACHE_DIR/build.hash)",
  "optimization": "$OPTIMIZATION_LEVEL",
  "sizes": {
    "js": "$JS_SIZE",
    "wasm": "$WASM_SIZE"
  }
}
EOF

# Build duration
BUILD_END=$(date +%s)
BUILD_DURATION=$((BUILD_END - BUILD_START))

echo ""
echo -e "${GREEN}✅ Build completed successfully!${NC}"
echo "================================================="
echo -e "  Duration:    ${BUILD_DURATION}s"
echo -e "  JS size:     $JS_SIZE"
echo -e "  WASM size:   $WASM_SIZE"
echo -e "  Output:      $OUTPUT_DIR"
echo -e "  Cache:       $CACHE_DIR"
echo "================================================="
echo ""
echo -e "${GREEN}💡 Tip: Run './build_optimized.sh' again to use cached build${NC}"
