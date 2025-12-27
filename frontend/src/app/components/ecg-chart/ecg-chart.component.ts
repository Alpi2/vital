import {
  Component,
  Input,
  OnInit,
  OnDestroy,
  ElementRef,
  ViewChild,
  AfterViewInit,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ECGDataPoint } from '../../core/services/ecg-data.service';

@Component({
  selector: 'app-ecg-chart',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="relative bg-black rounded-lg p-2">
      <div class="absolute top-2 left-4 z-10">
        <span class="text-xs text-gray-400">Lead II</span>
        <span class="ml-4 text-xs" [class]="getStatusColor()"> ● {{ getStatusText() }} </span>
      </div>
      <canvas #ecgCanvas class="w-full h-64 rounded" [class.alert-pulse]="hasAnomaly"> </canvas>
      <div class="absolute bottom-2 right-4 text-xs text-gray-500">
        {{ data.length }} samples @ {{ sampleRate }}Hz
      </div>
    </div>
  `,
  styles: [
    `
      :host {
        display: block;
      }
      canvas {
        background: linear-gradient(90deg, rgba(0, 0, 0, 0.1) 1px, transparent 1px),
          linear-gradient(rgba(0, 0, 0, 0.1) 1px, transparent 1px);
        background-size: 20px 20px;
      }
    `,
  ],
})
export class EcgChartComponent implements OnInit, OnDestroy, AfterViewInit {
  @Input() data: ECGDataPoint[] = [];
  @Input() isPlaying = false;

  @ViewChild('ecgCanvas') canvasRef!: ElementRef<HTMLCanvasElement>;

  private ctx!: CanvasRenderingContext2D;
  private animationId?: number;
  public sampleRate = 360;
  private lastRenderTime = 0;
  private readonly FPS = 60;
  private readonly frameTime = 1000 / this.FPS;

  public hasAnomaly = false;
  private gridColor = '#1a2b3c';
  private lineColor = '#10b981';
  private anomalyColor = '#ef4444';

  ngAfterViewInit(): void {
    const canvas = this.canvasRef.nativeElement;
    this.ctx = canvas.getContext('2d')!;

    // Canvas boyutlarını ayarla
    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());

    // Grid çiz
    this.drawGrid();

    // Render loop başlat
    this.render();
  }

  ngOnInit(): void {
    // Data değişikliklerini dinle
    // (Angular Signals ile otomatik olarak tetiklenecek)
  }

  ngOnDestroy(): void {
    window.cancelAnimationFrame(this.animationId!);
    window.removeEventListener('resize', () => this.resizeCanvas());
  }

  private resizeCanvas(): void {
    const canvas = this.canvasRef.nativeElement;
    const container = canvas.parentElement;

    if (container) {
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      this.drawGrid();
    }
  }

  private drawGrid(): void {
    const canvas = this.canvasRef.nativeElement;
    const width = canvas.width;
    const height = canvas.height;

    this.ctx.clearRect(0, 0, width, height);

    // Grid çiz
    this.ctx.strokeStyle = this.gridColor;
    this.ctx.lineWidth = 0.5;

    // Dikey çizgiler (her 20px'de bir)
    for (let x = 0; x <= width; x += 20) {
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, height);
      this.ctx.stroke();
    }

    // Yatay çizgiler (her 20px'de bir)
    for (let y = 0; y <= height; y += 20) {
      this.ctx.beginPath();
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(width, y);
      this.ctx.stroke();
    }
  }

  private render = (timestamp: number = 0): void => {
    if (timestamp - this.lastRenderTime >= this.frameTime) {
      this.drawECG();
      this.lastRenderTime = timestamp;
    }

    this.animationId = window.requestAnimationFrame(this.render);
  };

  private drawECG(): void {
    if (this.data.length < 2) return;

    const canvas = this.canvasRef.nativeElement;
    const width = canvas.width;
    const height = canvas.height;

    // Grid'i temizle
    this.drawGrid();

    // EKG çizgisi
    this.ctx.beginPath();
    this.ctx.strokeStyle = this.hasAnomaly ? this.anomalyColor : this.lineColor;
    this.ctx.lineWidth = 2;
    this.ctx.lineJoin = 'round';
    this.ctx.lineCap = 'round';

    const maxDataPoints = Math.min(this.data.length, width);
    const startIndex = Math.max(0, this.data.length - maxDataPoints);

    // Y ekseni scaling
    const values = this.data.slice(startIndex).map((d) => d.value);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const range = maxVal - minVal || 1;

    // X ekseni scaling (time-based)
    const timeRange = this.data[this.data.length - 1].timestamp - this.data[startIndex].timestamp;
    const pixelsPerMs = width / Math.max(timeRange, 1);

    // Çizimi başlat
    const startX = 0;
    const startY = height - ((this.data[startIndex].value - minVal) / range) * height;
    this.ctx.moveTo(startX, startY);

    // Her data point için çiz
    for (let i = 1; i < maxDataPoints; i++) {
      const dataIndex = startIndex + i;
      const dataPoint = this.data[dataIndex];

      // Anomali kontrolü
      if (dataPoint.anomaly) {
        this.hasAnomaly = true;
        this.ctx.strokeStyle = this.anomalyColor;
        this.ctx.stroke(); // Mevcut çizgiyi tamamla

        // Yeni çizgi başlat
        this.ctx.beginPath();
        this.ctx.moveTo(
          (dataPoint.timestamp - this.data[startIndex].timestamp) * pixelsPerMs,
          height - ((dataPoint.value - minVal) / range) * height
        );
      } else {
        this.hasAnomaly = false;
        if (this.ctx.strokeStyle !== this.lineColor) {
          this.ctx.strokeStyle = this.lineColor;
          this.ctx.stroke();
          this.ctx.beginPath();
          this.ctx.moveTo(
            (dataPoint.timestamp - this.data[startIndex].timestamp) * pixelsPerMs,
            height - ((dataPoint.value - minVal) / range) * height
          );
        }
      }

      const x = (dataPoint.timestamp - this.data[startIndex].timestamp) * pixelsPerMs;
      const y = height - ((dataPoint.value - minVal) / range) * height;

      this.ctx.lineTo(x, y);
    }

    this.ctx.stroke();

    // Real-time cursor (son noktada)
    if (this.data.length > 0) {
      const lastPoint = this.data[this.data.length - 1];
      const x = width - 10;
      const y = height - ((lastPoint.value - minVal) / range) * height;

      this.ctx.beginPath();
      this.ctx.arc(x, y, 4, 0, Math.PI * 2);
      this.ctx.fillStyle = this.hasAnomaly ? this.anomalyColor : this.lineColor;
      this.ctx.fill();

      // BPM değerini göster
      this.ctx.fillStyle = '#ffffff';
      this.ctx.font = '12px monospace';
      this.ctx.fillText(`${lastPoint.bpm.toFixed(0)} BPM`, x + 10, y - 10);
    }
  }

  getStatusColor(): string {
    return this.hasAnomaly ? 'text-red-400' : 'text-green-400';
  }

  getStatusText(): string {
    return this.hasAnomaly ? 'ANOMALY DETECTED' : 'NORMAL SINUS RHYTHM';
  }
}
