#!/bin/bash
set -euo pipefail

echo "Building VitalStream WebAssembly module..."

# Build directory oluştur
mkdir -p ../frontend/src/assets/wasm

# C++ kodunu derle
emcc \
    -O3 \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s USE_ES6_IMPORT_META=0 \
    -s ENVIRONMENT=web \
    -s EXPORTED_RUNTIME_METHODS='["ccall", "cwrap"]' \
    -s EXPORTED_FUNCTIONS='["_malloc", "_free"]' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MAXIMUM_MEMORY=256MB \
    -s ASSERTIONS=1 \
    --bind \
    -o ../frontend/src/assets/wasm/vitalstream.js \
    core/src/bindings.cpp \
    core/src/ecg_generator.cpp \
    core/src/ecg_analyzer.cpp \
    -Icore/include

echo "Build completed! Output: frontend/src/assets/wasm/"
