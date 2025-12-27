import { Injectable, signal, computed, effect } from '@angular/core';
import { BehaviorSubject, interval, Subscription, Observable } from 'rxjs';
import { WasmLoaderService } from './wasm-loader.service';

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

  constructor(private wasmLoader: WasmLoaderService) {
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
    const intervalMs = 1000 / (this.sampleRate() * this.playbackSpeed());
    this.playbackSubscription = interval(intervalMs).subscribe(() => {
      this.generateDataPoint();
    });
  }

  stopSimulation(): void {
    this.isPlaying.set(false);
    this.playbackSubscription?.unsubscribe();
  }

  private generateDataPoint(): void {
    this.wasmLoader.getModule().subscribe(module => {
      if (!module) return;

      const generator = new module.ECGGenerator(this.sampleRate(), 72);
      const analyzer = new module.ECGAnalyzer(this.sampleRate());

      const samples = generator.generateSamples(1);
      const analysis = analyzer.analyze(samples);

      const dataPoint: ECGDataPoint = {
        timestamp: Date.now(),
        value: samples[0],
        bpm: analysis.heartRate,
        anomaly: analysis.detectedAnomaly !== 0 ? this.getAnomalyName(analysis.detectedAnomaly) : null
      };

      this.dataBuffer.update(buffer => [...buffer, dataPoint]);
      this.dataStream$.next(dataPoint);

      if (dataPoint.anomaly) {
        this.anomalyDetected$.next(dataPoint.anomaly);
        this.triggerAlert(dataPoint.anomaly);
      }
    });
  }

  private calculateStatistics(): ECGStatistics {
    const buffer = this.dataBuffer();
    if (buffer.length === 0) {
      return { min: 0, max: 0, avg: 0, currentBPM: 0, anomalyCount: 0, lastAnomaly: null };
    }
    const values = buffer.map(d => d.value);
    const bpms = buffer.map(d => d.bpm);
    const anomalies = buffer.filter(d => d.anomaly !== null);
    return {
      min: Math.min(...values),
      max: Math.max(...values),
      avg: values.reduce((a, b) => a + b, 0) / values.length,
      currentBPM: bpms[bpms.length - 1] || 0,
      anomalyCount: anomalies.length,
      lastAnomaly: anomalies.length > 0 ? anomalies[anomalies.length - 1].anomaly : null
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
    // TODO: implement backend logging
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
}
