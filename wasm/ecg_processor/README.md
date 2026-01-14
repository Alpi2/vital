# ecg_processor (WASI WASM)

## Build

```bash
rustup target add wasm32-wasi
cargo build --release --target wasm32-wasi
```

The output WASM binary will be located at:

```text
target/wasm32-wasi/release/ecg_processor.wasm

Notes for frontend integration:
- The wasm-pack web build produces JS and WASM into `frontend/src/assets/wasm/ecg_processor/pkg/` (import path: `/assets/wasm/ecg_processor/pkg/ecg_processor.js`).
- The legacy Emscripten build (build_optimized.sh) now copies `ecg_analyzer.js` and `ecg_analyzer.wasm` into `frontend/src/assets/wasm/ecg_processor/` so the Angular assets path `/assets/wasm/ecg_processor/` serves all variants.
```
