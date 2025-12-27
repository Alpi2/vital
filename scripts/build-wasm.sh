#!/usr/bin/env bash
set -euo pipefail

echo "Building vital C++ core to WebAssembly (emscripten)"

# Ensure output dir
OUT_DIR=/src/out
mkdir -p "$OUT_DIR"

# Compile the core source directly to a JS/WASM module.
# Adjust include path if your headers are in a different location.
emcc /src/core/src/core.cpp \
  -I/src/core/include \
  -O3 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME="createVitalModule" \
  -s WASM=1 \
  -o "$OUT_DIR/vitalcore.js"

echo "Build complete: $OUT_DIR/vitalcore.js and $OUT_DIR/vitalcore.wasm"
