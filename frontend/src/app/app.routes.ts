import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'dashboard',
    pathMatch: 'full'
  },
  {
    path: 'auth',
    loadChildren: () => import('./features/auth/auth.routes')
      .then(m => m.AUTH_ROUTES)
  },
  {
    path: 'dashboard',
    loadComponent: () => import('./features/dashboard/dashboard.component')
      .then(m => m.DashboardComponent)
  },
  {
    path: 'security',
    loadChildren: () => import('./features/security/security.routes')
      .then(m => m.SECURITY_ROUTES)
  },
  {
    path: 'devices',
    loadChildren: () => import('./features/devices/devices.routes')
      .then(m => m.DEVICES_ROUTES)
  },
  {
    path: 'ai-analysis',
    loadChildren: () => import('./features/ai-analysis/ai-analysis.routes')
      .then(m => m.AI_ANALYSIS_ROUTES)
  },
  {
    path: 'monitoring',
    loadChildren: () => import('./features/monitoring/monitoring.routes')
      .then(m => m.MONITORING_ROUTES)
  },
  {
    path: 'alarms',
    loadChildren: () => import('./features/alarms/alarms.routes')
      .then(m => m.ALARMS_ROUTES)
  },
  {
    path: 'integrations',
    loadChildren: () => import('./features/integrations/integrations.routes')
      .then(m => m.INTEGRATIONS_ROUTES)
  },
  {
    path: 'clinical-modules',
    loadChildren: () => import('./features/clinical-modules/clinical-modules.routes')
      .then(m => m.CLINICAL_MODULES_ROUTES)
  },
  {
    path: 'admin',
    loadChildren: () => import('./features/admin/admin.routes')
      .then(m => m.ADMIN_ROUTES)
  },
  {
    path: 'research',
    loadChildren: () => import('./features/research/research.routes')
      .then(m => m.RESEARCH_ROUTES)
  },
  {
    path: 'training',
    loadChildren: () => import('./features/training/training.routes')
      .then(m => m.TRAINING_ROUTES)
  },
  {
    path: 'reports',
    loadChildren: () => import('./features/reports/reports.routes')
      .then(m => m.REPORTS_ROUTES)
  },
  {
    path: 'patients',
    loadChildren: () => import('./features/patients/patients.routes')
      .then(m => m.PATIENTS_ROUTES)
  },
  { path: '**', redirectTo: '' },
];
