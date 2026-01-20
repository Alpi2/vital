/**
 * WCAG 2.1 AA Accessibility Testing
 * Automated accessibility compliance checking
 */

import * as axe from 'axe-core';

export interface AccessibilityResult {
  violations: Violation[];
  passes: Pass[];
  incomplete: Incomplete[];
  totalIssues: number;
  criticalIssues: number;
  wcagLevel: 'A' | 'AA' | 'AAA';
  compliant: boolean;
}

export interface Violation {
  id: string;
  impact: 'critical' | 'serious' | 'moderate' | 'minor';
  description: string;
  help: string;
  helpUrl: string;
  nodes: ViolationNode[];
}

export interface ViolationNode {
  html: string;
  target: string[];
  failureSummary: string;
}

export interface Pass {
  id: string;
  description: string;
}

export interface Incomplete {
  id: string;
  description: string;
}

export class WCAGChecker {
  private wcagLevel: 'A' | 'AA' | 'AAA';

  constructor(wcagLevel: 'A' | 'AA' | 'AAA' = 'AA') {
    this.wcagLevel = wcagLevel;
  }

  async checkPage(url?: string): Promise<AccessibilityResult> {
    console.log(`Running WCAG ${this.wcagLevel} accessibility check...`);

    const results = await axe.run({
      runOnly: {
        type: 'tag',
        values: [`wcag2${this.wcagLevel.toLowerCase()}`, 'best-practice']
      }
    });

    const violations: Violation[] = results.violations.map(v => ({
      id: v.id,
      impact: v.impact as any,
      description: v.description,
      help: v.help,
      helpUrl: v.helpUrl,
      nodes: v.nodes.map(n => ({
        html: n.html,
        target: n.target,
        failureSummary: n.failureSummary || ''
      }))
    }));

    const passes: Pass[] = results.passes.map(p => ({
      id: p.id,
      description: p.description
    }));

    const incomplete: Incomplete[] = results.incomplete.map(i => ({
      id: i.id,
      description: i.description
    }));

    const criticalIssues = violations.filter(
      v => v.impact === 'critical' || v.impact === 'serious'
    ).length;

    return {
      violations,
      passes,
      incomplete,
      totalIssues: violations.length,
      criticalIssues,
      wcagLevel: this.wcagLevel,
      compliant: violations.length === 0
    };
  }

  async checkColorContrast(): Promise<boolean> {
    const results = await axe.run({
      runOnly: {
        type: 'rule',
        values: ['color-contrast']
      }
    });

    return results.violations.length === 0;
  }

  async checkKeyboardNavigation(): Promise<boolean> {
    const results = await axe.run({
      runOnly: {
        type: 'rule',
        values: ['keyboard', 'focus-order-semantics']
      }
    });

    return results.violations.length === 0;
  }

  async checkAriaLabels(): Promise<boolean> {
    const results = await axe.run({
      runOnly: {
        type: 'rule',
        values: ['aria-allowed-attr', 'aria-required-attr', 'aria-valid-attr']
      }
    });

    return results.violations.length === 0;
  }

  async checkFormLabels(): Promise<boolean> {
    const results = await axe.run({
      runOnly: {
        type: 'rule',
        values: ['label', 'label-title-only']
      }
    });

    return results.violations.length === 0;
  }

  async checkImageAltText(): Promise<boolean> {
    const results = await axe.run({
      runOnly: {
        type: 'rule',
        values: ['image-alt']
      }
    });

    return results.violations.length === 0;
  }

  printReport(result: AccessibilityResult): void {
    console.log('\n=== WCAG Accessibility Report ===');
    console.log(`WCAG Level: ${result.wcagLevel}`);
    console.log(`Compliant: ${result.compliant ? '✓ YES' : '✗ NO'}`);
    console.log(`Total Issues: ${result.totalIssues}`);
    console.log(`Critical Issues: ${result.criticalIssues}`);
    console.log(`Passed Checks: ${result.passes.length}`);
    console.log(`Incomplete Checks: ${result.incomplete.length}`);

    if (result.violations.length > 0) {
      console.log('\nViolations:');
      result.violations.forEach((violation, index) => {
        console.log(`\n${index + 1}. [${violation.impact.toUpperCase()}] ${violation.id}`);
        console.log(`   ${violation.description}`);
        console.log(`   Help: ${violation.help}`);
        console.log(`   More info: ${violation.helpUrl}`);
        console.log(`   Affected elements: ${violation.nodes.length}`);
        
        violation.nodes.slice(0, 3).forEach((node, nodeIndex) => {
          console.log(`     ${nodeIndex + 1}. ${node.target.join(' > ')}`);
        });
      });
    }

    if (result.incomplete.length > 0) {
      console.log('\nIncomplete Checks (Manual Review Required):');
      result.incomplete.forEach((item, index) => {
        console.log(`  ${index + 1}. ${item.id}: ${item.description}`);
      });
    }
  }
}

// Specific accessibility tests for VitalStream
export class VitalStreamAccessibilityTests {
  private checker: WCAGChecker;

  constructor() {
    this.checker = new WCAGChecker('AA');
  }

  async testPatientDashboard(): Promise<boolean> {
    console.log('Testing Patient Dashboard accessibility...');
    const result = await this.checker.checkPage();
    this.checker.printReport(result);
    return result.compliant;
  }

  async testAlarmNotifications(): Promise<boolean> {
    console.log('Testing Alarm Notifications accessibility...');
    
    // Check for proper ARIA live regions
    const alarmRegions = document.querySelectorAll('[role="alert"]');
    if (alarmRegions.length === 0) {
      console.error('No ARIA alert regions found for alarms');
      return false;
    }

    // Check for screen reader announcements
    const hasAriaLive = Array.from(alarmRegions).every(
      el => el.getAttribute('aria-live') === 'assertive'
    );

    return hasAriaLive;
  }

  async testWaveformVisualization(): Promise<boolean> {
    console.log('Testing Waveform Visualization accessibility...');
    
    // Waveforms should have text alternatives
    const waveforms = document.querySelectorAll('[data-cy="waveform-canvas"]');
    
    for (const waveform of Array.from(waveforms)) {
      const hasAltText = waveform.getAttribute('aria-label') !== null;
      const hasDescription = waveform.getAttribute('aria-describedby') !== null;
      
      if (!hasAltText && !hasDescription) {
        console.error('Waveform missing accessibility labels');
        return false;
      }
    }

    return true;
  }

  async testKeyboardNavigation(): Promise<boolean> {
    console.log('Testing Keyboard Navigation...');
    return await this.checker.checkKeyboardNavigation();
  }

  async testColorContrast(): Promise<boolean> {
    console.log('Testing Color Contrast...');
    return await this.checker.checkColorContrast();
  }

  async runAllTests(): Promise<{ passed: number; failed: number; total: number }> {
    const tests = [
      { name: 'Patient Dashboard', fn: () => this.testPatientDashboard() },
      { name: 'Alarm Notifications', fn: () => this.testAlarmNotifications() },
      { name: 'Waveform Visualization', fn: () => this.testWaveformVisualization() },
      { name: 'Keyboard Navigation', fn: () => this.testKeyboardNavigation() },
      { name: 'Color Contrast', fn: () => this.testColorContrast() }
    ];

    let passed = 0;
    let failed = 0;

    for (const test of tests) {
      try {
        const result = await test.fn();
        if (result) {
          passed++;
          console.log(`✓ ${test.name} passed`);
        } else {
          failed++;
          console.log(`✗ ${test.name} failed`);
        }
      } catch (error) {
        failed++;
        console.log(`✗ ${test.name} failed with error:`, error);
      }
    }

    console.log(`\n=== Accessibility Test Summary ===`);
    console.log(`Passed: ${passed}/${tests.length}`);
    console.log(`Failed: ${failed}/${tests.length}`);

    return { passed, failed, total: tests.length };
  }
}
