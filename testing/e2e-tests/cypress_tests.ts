/**
 * End-to-End Test Automation with Cypress
 * Tests complete user workflows
 */

describe('VitalStream E2E Tests', () => {
  beforeEach(() => {
    cy.visit('http://localhost:4200');
    cy.login('test@vitalstream.com', 'password123');
  });

  describe('Patient Monitoring Workflow', () => {
    it('should display patient list', () => {
      cy.get('[data-cy=patient-list]').should('be.visible');
      cy.get('[data-cy=patient-card]').should('have.length.greaterThan', 0);
    });

    it('should navigate to patient detail', () => {
      cy.get('[data-cy=patient-card]').first().click();
      cy.url().should('include', '/patient/');
      cy.get('[data-cy=patient-vitals]').should('be.visible');
    });

    it('should display real-time waveforms', () => {
      cy.get('[data-cy=patient-card]').first().click();
      cy.get('[data-cy=ecg-waveform]').should('be.visible');
      cy.get('[data-cy=waveform-canvas]').should('exist');
    });

    it('should show vital signs values', () => {
      cy.get('[data-cy=patient-card]').first().click();
      cy.get('[data-cy=heart-rate]').should('contain.text', 'bpm');
      cy.get('[data-cy=spo2]').should('contain.text', '%');
      cy.get('[data-cy=blood-pressure]').should('be.visible');
    });
  });

  describe('Alarm Management', () => {
    it('should trigger alarm on abnormal vitals', () => {
      cy.simulateAbnormalVitals({ heartRate: 45 });
      cy.get('[data-cy=alarm-notification]').should('be.visible');
      cy.get('[data-cy=alarm-level]').should('contain.text', 'CRITICAL');
    });

    it('should acknowledge alarm', () => {
      cy.simulateAbnormalVitals({ heartRate: 45 });
      cy.get('[data-cy=alarm-notification]').should('be.visible');
      cy.get('[data-cy=acknowledge-alarm]').click();
      cy.get('[data-cy=alarm-acknowledged]').should('be.visible');
    });

    it('should display alarm history', () => {
      cy.get('[data-cy=alarm-history-btn]').click();
      cy.get('[data-cy=alarm-history-list]').should('be.visible');
      cy.get('[data-cy=alarm-history-item]').should('have.length.greaterThan', 0);
    });
  });

  describe('Report Generation', () => {
    it('should generate PDF report', () => {
      cy.get('[data-cy=patient-card]').first().click();
      cy.get('[data-cy=generate-report]').click();
      cy.get('[data-cy=report-type]').select('24-hour-summary');
      cy.get('[data-cy=generate-pdf]').click();
      cy.wait(2000);
      cy.get('[data-cy=download-report]').should('be.visible');
    });

    it('should export data to CSV', () => {
      cy.get('[data-cy=patient-card]').first().click();
      cy.get('[data-cy=export-data]').click();
      cy.get('[data-cy=export-format]').select('csv');
      cy.get('[data-cy=export-btn]').click();
      cy.readFile('cypress/downloads/vitals-export.csv').should('exist');
    });
  });

  describe('Settings and Preferences', () => {
    it('should update alarm thresholds', () => {
      cy.get('[data-cy=settings-btn]').click();
      cy.get('[data-cy=alarm-settings]').click();
      cy.get('[data-cy=hr-low-threshold]').clear().type('50');
      cy.get('[data-cy=hr-high-threshold]').clear().type('120');
      cy.get('[data-cy=save-settings]').click();
      cy.get('[data-cy=success-message]').should('be.visible');
    });

    it('should toggle dark mode', () => {
      cy.get('[data-cy=settings-btn]').click();
      cy.get('[data-cy=dark-mode-toggle]').click();
      cy.get('body').should('have.class', 'dark-theme');
    });
  });

  describe('Multi-Patient View', () => {
    it('should display multiple patients in grid', () => {
      cy.get('[data-cy=grid-view]').click();
      cy.get('[data-cy=patient-grid]').should('be.visible');
      cy.get('[data-cy=patient-tile]').should('have.length.greaterThan', 4);
    });

    it('should filter patients by department', () => {
      cy.get('[data-cy=department-filter]').select('ICU');
      cy.get('[data-cy=patient-card]').each(($card) => {
        cy.wrap($card).should('contain.text', 'ICU');
      });
    });
  });

  describe('Performance Tests', () => {
    it('should load patient list within 2 seconds', () => {
      const start = Date.now();
      cy.visit('http://localhost:4200/patients');
      cy.get('[data-cy=patient-list]').should('be.visible').then(() => {
        const loadTime = Date.now() - start;
        expect(loadTime).to.be.lessThan(2000);
      });
    });

    it('should render waveforms at 60 FPS', () => {
      cy.get('[data-cy=patient-card]').first().click();
      cy.window().then((win) => {
        let frameCount = 0;
        const startTime = performance.now();
        
        const countFrames = () => {
          frameCount++;
          const elapsed = performance.now() - startTime;
          if (elapsed < 1000) {
            win.requestAnimationFrame(countFrames);
          } else {
            const fps = frameCount / (elapsed / 1000);
            expect(fps).to.be.greaterThan(55); // Allow some variance
          }
        };
        
        win.requestAnimationFrame(countFrames);
      });
    });
  });

  describe('Accessibility Tests', () => {
    it('should be keyboard navigable', () => {
      cy.get('body').tab();
      cy.focused().should('have.attr', 'data-cy', 'patient-list');
      cy.focused().tab();
      cy.focused().should('have.attr', 'data-cy', 'patient-card');
    });

    it('should have proper ARIA labels', () => {
      cy.get('[data-cy=patient-card]').first().should('have.attr', 'aria-label');
      cy.get('[data-cy=alarm-notification]').should('have.attr', 'role', 'alert');
    });

    it('should meet WCAG 2.1 AA contrast requirements', () => {
      cy.injectAxe();
      cy.checkA11y(null, {
        rules: {
          'color-contrast': { enabled: true }
        }
      });
    });
  });
});

// Custom Cypress commands
Cypress.Commands.add('login', (email: string, password: string) => {
  cy.get('[data-cy=email-input]').type(email);
  cy.get('[data-cy=password-input]').type(password);
  cy.get('[data-cy=login-btn]').click();
  cy.get('[data-cy=dashboard]').should('be.visible');
});

Cypress.Commands.add('simulateAbnormalVitals', (vitals: any) => {
  cy.window().then((win) => {
    (win as any).simulateVitals(vitals);
  });
});

declare global {
  namespace Cypress {
    interface Chainable {
      login(email: string, password: string): Chainable<void>;
      simulateAbnormalVitals(vitals: any): Chainable<void>;
    }
  }
}
