import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';

export interface AnomalyLog {
  anomaly_type: string;
  severity: 'low' | 'medium' | 'high';
  confidence: number;
  timestamp: string;
  bpm_at_detection?: number;
  details?: string;
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  private apiUrl = environment.apiUrl;
  
  constructor(private http: HttpClient) {}
  
  // Generic HTTP methods
  get<T>(endpoint: string): Observable<T> {
    return this.http.get<T>(`${this.apiUrl}${endpoint}`);
  }
  
  post<T>(endpoint: string, data: any): Observable<T> {
    return this.http.post<T>(`${this.apiUrl}${endpoint}`, data);
  }
  
  put<T>(endpoint: string, data: any): Observable<T> {
    return this.http.put<T>(`${this.apiUrl}${endpoint}`, data);
  }
  
  delete<T>(endpoint: string): Observable<T> {
    return this.http.delete<T>(`${this.apiUrl}${endpoint}`);
  }
  
  logAnomaly(patientId: number, anomaly: Omit<AnomalyLog, 'patient_id'>): Observable<any> {
    const payload = {
      ...anomaly,
      patient_id: patientId,
      session_id: this.getCurrentSessionId()
    };
    
    return this.http.post(`${this.apiUrl}/anomalies/`, payload);
  }
  
  getAnomalyStats(patientId?: number): Observable<any> {
    const params: any = {};
    if (patientId) params.patient_id = patientId;
    
    return this.http.get(`${this.apiUrl}/anomalies/stats`, { params });
  }
  
  generateReport(patientId: number): Observable<Blob> {
    return this.http.get(
      `${this.apiUrl}/reports/patient/${patientId}`,
      { responseType: 'blob' }
    );
  }
  
  private getCurrentSessionId(): string {
    // Generate or retrieve current session ID
    const sessionId = localStorage.getItem('ecg_session_id');
    if (sessionId) return sessionId;
    
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem('ecg_session_id', newSessionId);
    return newSessionId;
  }
}
