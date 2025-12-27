import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [CommonModule],
  template: `
    <header class="w-full bg-gray-800 p-4 flex items-center justify-between">
      <div class="flex items-center gap-4">
        <h1 class="text-lg font-bold">VitalStream</h1>
        <div class="text-sm text-gray-400">{{ patient?.name }}</div>
      </div>
      <div class="flex items-center gap-3">
        <div [class.text-green-400]="isConnected">●</div>
        <button class="btn btn-sm" (click)="$emitToggle()">
          {{ isConnected ? 'Disconnect' : 'Connect' }}
        </button>
      </div>
    </header>
  `,
})
export class HeaderComponent {
  @Input() patient: any;
  @Input() isConnected = true;
  @Output() toggleConnect = new EventEmitter<void>();

  $emitToggle() {
    this.toggleConnect.emit();
  }
}
