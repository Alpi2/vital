import { TestBed } from '@angular/core/testing';
import { ECGDataService } from './ecg-data.service';

describe('ECGDataService', () => {
  let service: ECGDataService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ECGDataService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should start and stop simulation', () => {
    service.startSimulation();
    expect(service['isPlaying']()).toBe(true);
    
    service.stopSimulation();
    expect(service['isPlaying']()).toBe(false);
  });

  it('should clear data buffer', () => {
    service.clearBuffer();
    expect(service.currentData().length).toBe(0);
  });
});
