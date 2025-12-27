import { Injectable, signal, effect } from '@angular/core';
import { BehaviorSubject, Observable, throwError } from 'rxjs';
import { catchError, shareReplay } from 'rxjs/operators';

declare const Module: any;

interface WasmModule {
  ECGGenerator: new (sampleRate: number, bpm: number) => any;
  ECGAnalyzer: new (sampleRate: number) => any;
  AnalysisResult: any;
  AnomalyType: any;
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
      const modulePath = '/assets/wasm/vitalstream.js';
      let moduleFactory: any = null;

      try {
        const resp = await fetch(modulePath);
        if (!resp.ok) throw new Error(`WASM asset not found: ${resp.status}`);
        const jsText = await resp.text();
        const blob = new Blob([jsText], { type: 'text/javascript' });
        const blobUrl = URL.createObjectURL(blob);
        const wasmMod = await import(/* @vite-ignore */ blobUrl);
        moduleFactory = wasmMod && wasmMod.default ? wasmMod.default : wasmMod;
        URL.revokeObjectURL(blobUrl);
      } catch (fetchErr) {
        // Fallback: attempt runtime import directly (may still fail in dev optimizer)
        const wasmMod = await import(/* @vite-ignore */ modulePath);
        moduleFactory = wasmMod && wasmMod.default ? wasmMod.default : wasmMod;
      }

      if (!moduleFactory) throw new Error('WASM module factory unavailable');

      const module = await moduleFactory({
        locateFile: (path: string) => (path.endsWith('.wasm') ? `/assets/wasm/${path}` : path),
      });

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
