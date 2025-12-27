import { Component, OnInit, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { HeaderComponent } from '../../components/header/header.component';
import { SidebarComponent } from '../../components/sidebar/sidebar.component';
import { EcgChartComponent } from '../../components/ecg-chart/ecg-chart.component';
import { ECGDataService } from '../../core/services/ecg-data.service';

@Component({
  selector: 'app-main-layout',
  standalone: true,
  imports: [CommonModule, RouterModule, HeaderComponent, SidebarComponent, EcgChartComponent],
  template: `
    <div class="min-h-screen bg-gray-900 text-gray-100">
      <app-header
        [patient]="currentPatient"
        [isConnected]="isConnected()"
        (toggleConnect)="toggleConnection()"
      >
      </app-header>

      <div class="flex">
        <app-sidebar [statistics]="statistics()" [alerts]="alerts()" (clearAlerts)="clearAlerts()">
        </app-sidebar>

        <main class="flex-1 p-6">
          <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Real-time ECG Chart -->
            <div class="lg:col-span-2">
              <div class="bg-gray-800 rounded-xl p-4 shadow-2xl border border-gray-700">
                <div class="flex justify-between items-center mb-4">
                  <h2 class="text-xl font-bold text-medical-green">
                    <span class="animate-pulse">●</span> Real-time ECG Monitor
                  </h2>
                  <div class="flex gap-2">
                    <button class="btn btn-primary" (click)="togglePlayback()">
                      {{ isPlaying() ? '⏸️ Pause' : '▶️ Start' }}
                    </button>
                    <button class="btn btn-secondary" (click)="clearData()">🗑️ Clear</button>
                  </div>
                </div>
                <app-ecg-chart [data]="ecgData.currentData()" [isPlaying]="isPlaying()">
                </app-ecg-chart>
              </div>
            </div>

            <!-- Vital Stats -->
            <div class="space-y-6">
              <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
                <h3 class="text-sm font-semibold">Vital Stats</h3>
                <pre class="text-xs mt-2">{{ statistics() | json }}</pre>
              </div>

              <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
                <h3 class="text-sm font-semibold">Anomaly Alerts</h3>
                <ul class="text-sm mt-2 space-y-1">
                  <li *ngFor="let a of alerts()">{{ a.type }} — {{ a.message }}</li>
                  <li *ngIf="alerts().length === 0" class="text-gray-500">No alerts</li>
                </ul>
              </div>

              <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
                <h3 class="text-sm font-semibold">Controls</h3>
                <div class="mt-2">
                  <label class="text-xs">Playback speed: {{ playbackSpeed() }}</label>
                  <input
                    type="range"
                    min="0.5"
                    max="2"
                    step="0.1"
                    [value]="playbackSpeed()"
                    (input)="changeSpeed($any($event).target.value)"
                  />
                </div>
                <div class="mt-2">
                  <button class="btn btn-secondary" (click)="injectAnomaly('PVC')">
                    Inject PVC
                  </button>
                </div>
              </div>
            </div>
          </div>

          <!-- Additional Charts Placeholder -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
              BPM Trend (coming soon)
            </div>
            <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
              Anomaly Distribution (coming soon)
            </div>
          </div>

          <!-- Patient Info Placeholder -->
          <div class="mt-6">
            <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
              Patient Info & Reports (coming soon)
            </div>
          </div>
        </main>
      </div>
    </div>
  `,
  styles: [
    `
      .btn {
        @apply px-4 py-2 rounded-lg font-medium transition-all duration-200;
      }
      .btn-primary {
        @apply bg-medical-blue hover:bg-blue-600 text-white;
      }
      .btn-secondary {
        @apply bg-gray-700 hover:bg-gray-600 text-gray-200;
      }
      .btn-danger {
        @apply bg-red-600 hover:bg-red-700 text-white;
      }
    `,
  ],
})
export class MainLayoutComponent implements OnInit {
  currentPatient = {
    name: 'John Doe',
    age: 45,
    id: 'PT-2024-001',
    lastCheckup: '2024-01-15',
  };

  isConnected = signal(true);
  isPlaying = signal(false);
  playbackSpeed = signal(1.0);
  statistics = computed(() => this.ecgData.statistics());
  alerts = signal<any[]>([]);

  constructor(public ecgData: ECGDataService) {}

  ngOnInit(): void {
    // Anomali dinleyicisi
    this.ecgData.getAnomalyStream().subscribe((anomaly) => {
      if (anomaly) {
        this.alerts.update((alerts) => [
          {
            id: Date.now(),
            type: anomaly,
            timestamp: new Date(),
            message: `${anomaly} detected`,
            severity: this.getSeverity(anomaly),
          },
          ...alerts.slice(0, 9), // Son 10 alert
        ]);
      }
    });
  }

  toggleConnection(): void {
    this.isConnected.update((v) => !v);
    if (!this.isConnected()) {
      this.ecgData.stopSimulation();
      this.isPlaying.set(false);
    }
  }

  togglePlayback(): void {
    if (this.isPlaying()) {
      this.ecgData.stopSimulation();
    } else {
      this.ecgData.startSimulation();
    }
    this.isPlaying.update((v) => !v);
  }

  clearData(): void {
    this.ecgData.clearBuffer();
    this.alerts.set([]);
  }

  changeSpeed(speed: any): void {
    const val = Number(speed);
    this.playbackSpeed.set(val);
    this.ecgData.setPlaybackSpeed(val);
  }

  injectAnomaly(type: any): void {
    const t = String(type);
    console.log(`Injecting anomaly: ${t}`);
  }

  clearAlerts(): void {
    this.alerts.set([]);
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
