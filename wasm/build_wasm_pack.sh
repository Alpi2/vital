#!/bin/bash
set -e

ECG_DIR="ecg_processor"
OUT_DIR="../frontend/src/assets/wasm/ecg_processor"

echo "🔨 Building Rust WASM package with wasm-pack for ${ECG_DIR}..."
pushd "$ECG_DIR" > /dev/null

if ! command -v wasm-pack &> /dev/null; then
  echo "wasm-pack not found. Install with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
  exit 1
fi

wasm-pack build --release --target web --out-dir "$OUT_DIR/pkg"

popd > /dev/null

echo "✅ wasm-pack build finished. Output: $OUT_DIR/pkg"