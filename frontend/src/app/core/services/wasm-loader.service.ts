import { Injectable, signal, effect } from '@angular/core';
import { BehaviorSubject, Observable, throwError } from 'rxjs';
import { catchError, shareReplay } from 'rxjs/operators';

declare const Module: any;

interface WasmModule {
  ECGGenerator?: new (sampleRate: number, bpm: number) => any;
  ECGAnalyzer?: new (sampleRate: number) => any;
  analyze_ecg?: (input: Float64Array, samplingRate: number) => any;
  AnalysisResult?: any;
  AnomalyType?: any;
}

@Injectable({ providedIn: 'root' })
export class WasmLoaderService {
  private wasmModule$ = new BehaviorSubject<WasmModule | null>(null);
  private isLoading = signal(false);
  private error = signal<string | null>(null);

  constructor() {
    this.loadWasmModule();
  }

  private async loadWasmModule(): Promise<void> {
    this.isLoading.set(true);
    try {
      // Dynamic runtime import: fetch module text then import via blob URL
      // This avoids Vite/optimizer trying to resolve the absolute '/assets/..' path at build time.
      // New wasm-pack generated module path
      const pkgBase = '/assets/wasm/ecg_processor/pkg/ecg_processor.js';
      const wasmBase = '/assets/wasm/ecg_processor/pkg/ecg_processor_bg.wasm';

      // Try dynamic import via blob (avoid bundler resolving issues in dev)
      let imported: any = null;
      try {
        const resp = await fetch(`${pkgBase}?_=${Date.now()}`);
        if (!resp.ok) throw new Error(`WASM pkg not found: ${resp.status}`);
        const jsText = await resp.text();
        const blob = new Blob([jsText], { type: 'text/javascript' });
        const blobUrl = URL.createObjectURL(blob);
        imported = await import(/* @vite-ignore */ blobUrl);
        URL.revokeObjectURL(blobUrl);
      } catch (err) {
        // fallback to direct import
        imported = await import(/* @vite-ignore */ pkgBase);
      }

      if (!imported) throw new Error('WASM package import failed');

      // wasm-pack web target exports an init default and named exports
      const initFn = imported.default || imported.init || imported;
      if (typeof initFn !== 'function') {
        throw new Error('WASM package init function not found');
      }

      // Initialize with explicit wasm path
      await initFn(wasmBase);

      // Expose exports (ECGAnalyzer etc.)
      const module = imported;


      this.wasmModule$.next(module as WasmModule);
      this.error.set(null);
    } catch (err: any) {
      console.error('WASM load error:', err);
      this.error.set('Failed to load ECG analyzer module');
      this.wasmModule$.next(null);
    } finally {
      this.isLoading.set(false);
    }
  }

  getModule(): Observable<WasmModule | null> {
    return this.wasmModule$.asObservable().pipe(
      shareReplay(1),
      catchError((err) => {
        this.error.set(err?.message ?? String(err));
        return throwError(() => err);
      })
    );
  }

  get isLoadingState() {
    return this.isLoading.asReadonly();
  }

  get errorState() {
    return this.error.asReadonly();
  }
}
