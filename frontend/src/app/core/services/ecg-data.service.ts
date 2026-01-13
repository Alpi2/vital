import { Injectable, signal, computed, effect } from '@angular/core';
import { BehaviorSubject, interval, Subscription, Observable } from 'rxjs';
import { WasmLoaderService } from './wasm-loader.service';
import { ApiService } from './api.service';

export interface ECGDataPoint {
  timestamp: number;
  value: number;
  bpm: number;
  anomaly: string | null;
}

export interface ECGStatistics {
  min: number;
  max: number;
  avg: number;
  currentBPM: number;
  anomalyCount: number;
  lastAnomaly: string | null;
}

@Injectable({ providedIn: 'root' })
export class ECGDataService {
  private dataBuffer = signal<ECGDataPoint[]>([]);
  private isPlaying = signal(false);
  private playbackSpeed = signal(1.0);
  private sampleRate = signal(360);

  currentData = computed(() => this.dataBuffer());
  statistics = computed(() => this.calculateStatistics());

  private dataStream$ = new BehaviorSubject<ECGDataPoint | null>(null);
  private anomalyDetected$ = new BehaviorSubject<string | null>(null);

  private playbackSubscription?: Subscription;

  constructor(private wasmLoader: WasmLoaderService, private apiService: ApiService) {
    effect(() => {
      const buffer = this.dataBuffer();
      if (buffer.length > 1000) {
        this.dataBuffer.set(buffer.slice(-1000));
      }
    });
  }

  startSimulation(): void {
    if (this.isPlaying()) return;
    this.isPlaying.set(true);

    // Render at a reduced frame rate to match real-time display and reduce perceived speed
    const renderHz = 30; // lower FPS gives smoother, realistic playback
    const samplesPerTick = Math.max(
      1,
      Math.round((this.sampleRate() / renderHz) * this.playbackSpeed())
    );
    const intervalMs = 1000 / renderHz;

    console.debug('[ECG] startSimulation', {
      renderHz,
      samplesPerTick,
      sampleRate: this.sampleRate(),
      playbackSpeed: this.playbackSpeed(),
    });

    this.playbackSubscription = interval(intervalMs).subscribe(() => {
      this.generateDataPoints(samplesPerTick);
    });
  }

  stopSimulation(): void {
    this.isPlaying.set(false);
    this.playbackSubscription?.unsubscribe();
  }

  private generateDataPoint(): void {
    // Backwards-compatible: generate a single sample
    this.generateDataPoints(1);
  }

  private generateDataPoints(count: number): void {
    console.debug('[ECG] generateDataPoints', {
      count,
      sampleRate: this.sampleRate(),
      playbackSpeed: this.playbackSpeed(),
    });

    this.wasmLoader.getModule().subscribe((module) => {
      if (!module) {
        console.warn('[ECG] WASM module not loaded - skipping data generation');
        return;
      }

      try {
        const generator = new module.ECGGenerator(this.sampleRate(), 72);
        const analyzer = new module.ECGAnalyzer(this.sampleRate());

        // Request `count` contiguous samples and analyze them as a block
        const samples = generator.generateSamples(count);
        const analysis = analyzer.analyze(samples);

        // Spread timestamps over the samples so the chart has accurate sample timing
        const now = Date.now();
        const sampleIntervalMs = 1000 / this.sampleRate();
        const baseTime = now - (count - 1) * sampleIntervalMs;

        for (let i = 0; i < count; i++) {
          const dataPoint: ECGDataPoint = {
            timestamp: baseTime + i * sampleIntervalMs,
            value: samples[i],
            bpm: analysis.heartRate,
            anomaly:
              analysis.detectedAnomaly !== 0 ? this.getAnomalyName(analysis.detectedAnomaly) : null,
          };

          this.dataBuffer.update((buffer) => [...buffer, dataPoint]);
          this.dataStream$.next(dataPoint);

          if (dataPoint.anomaly) {
            this.anomalyDetected$.next(dataPoint.anomaly);
            this.triggerAlert(dataPoint.anomaly);
          }
        }

        console.debug('[ECG] generated points', {
          count,
          firstSample: samples[0],
          heartRate: analysis.heartRate,
          detectedAnomaly: analysis.detectedAnomaly,
        });
      } catch (err) {
        console.error('[ECG] error generating data points', err);
      }
    });
  }

  private calculateStatistics(): ECGStatistics {
    const buffer = this.dataBuffer();
    if (buffer.length === 0) {
      return { min: 0, max: 0, avg: 0, currentBPM: 0, anomalyCount: 0, lastAnomaly: null };
    }

    // Compute statistics over a moving window (last `windowSeconds`) to reduce noise
    const windowSeconds = 5; // average over last 5 seconds
    const windowSamples = Math.max(1, Math.round(this.sampleRate() * windowSeconds));
    const sliceStart = Math.max(0, buffer.length - windowSamples);
    const window = buffer.slice(sliceStart);

    const values = window.map((d) => d.value);
    const bpms = window.map((d) => d.bpm);
    const anomalies = window.filter((d) => d.anomaly !== null);

    const avg = values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;

    return {
      min: values.length ? Math.min(...values) : 0,
      max: values.length ? Math.max(...values) : 0,
      avg,
      currentBPM: bpms.length ? bpms[bpms.length - 1] : 0,
      anomalyCount: anomalies.length,
      lastAnomaly: anomalies.length > 0 ? anomalies[anomalies.length - 1].anomaly : null,
    };
  }

  private getAnomalyName(type: number): string {
    const anomalies = ['NONE', 'TACHYCARDIA', 'BRADYCARDIA', 'PVC', 'AFIB', 'NOISE_ARTIFACT'];
    return anomalies[type] || 'UNKNOWN';
  }

  private triggerAlert(anomaly: string): void {
    console.warn(`ALERT: ${anomaly} detected!`);
    this.logAnomalyToBackend(anomaly);
  }

  private logAnomalyToBackend(anomaly: string): void {
    const stats = this.statistics();
    this.apiService
      .logAnomaly(1, {
        // patientId dinamik olmalı
        anomaly_type: anomaly,
        severity: this.getSeverity(anomaly),
        confidence: 0.85,
        timestamp: new Date().toISOString(),
        bpm_at_detection: stats.currentBPM,
      })
      .subscribe();
  }

  clearBuffer(): void {
    this.dataBuffer.set([]);
  }

  setPlaybackSpeed(speed: number): void {
    this.playbackSpeed.set(speed);
    if (this.isPlaying()) {
      this.stopSimulation();
      this.startSimulation();
    }
  }

  getDataStream(): Observable<ECGDataPoint | null> {
    return this.dataStream$.asObservable();
  }

  getAnomalyStream(): Observable<string | null> {
    return this.anomalyDetected$.asObservable();
  }

  private getSeverity(anomaly: string): 'low' | 'medium' | 'high' {
    const severityMap: Record<string, 'low' | 'medium' | 'high'> = {
      TACHYCARDIA: 'high',
      BRADYCARDIA: 'high',
      AFIB: 'high',
      PVC: 'medium',
      NOISE_ARTIFACT: 'low',
    };
    return severityMap[anomaly] || 'medium';
  }
}
