import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [CommonModule],
  template: `
    <aside class="w-64 bg-gray-900 text-white p-4 h-full">
      <h2 class="font-semibold mb-2">Session</h2>
      <div class="text-sm text-gray-300 mb-4">Status: {{ statistics?.status || 'idle' }}</div>
      <h3 class="font-semibold">Alerts</h3>
      <ul class="text-sm mt-2">
        <li *ngFor="let a of alerts">{{ a }}</li>
      </ul>
      <button class="mt-4 btn btn-sm" (click)="$emitClear()">Clear Alerts</button>
    </aside>
  `,
})
export class SidebarComponent {
  @Input() statistics: any;
  @Input() alerts: string[] = [];
  @Output() clearAlerts = new EventEmitter<void>();

  $emitClear() {
    this.clearAlerts.emit();
  }
}
