//! Sepsis Risk Assessment
//!
//! Implements qSOFA (quick Sequential Organ Failure Assessment) and SIRS criteria.

use crate::{InferenceError, Result};

/// Sepsis risk assessment result
#[derive(Debug, Clone)]
pub struct SepsisRiskScore {
    pub qsofa_score: u8,
    pub sirs_score: u8,
    pub sepsis_risk: SepsisRisk,
    pub recommendation: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SepsisRisk {
    Low,
    Moderate,
    High,
}

/// Vital signs for sepsis assessment
#[derive(Debug, Clone)]
pub struct SepsisVitals {
    pub respiratory_rate: f32,
    pub systolic_bp: f32,
    pub mental_status_altered: bool,
    pub temperature: f32,
    pub heart_rate: f32,
    pub wbc_count: Option<f32>,  // White blood cell count (10^9/L)
}

pub struct SepsisRiskCalculator;

impl SepsisRiskCalculator {
    /// Calculate qSOFA score
    ///
    /// qSOFA criteria:
    /// - Respiratory rate ≥ 22/min
    /// - Altered mental status
    /// - Systolic BP ≤ 100 mmHg
    ///
    /// Score ≥ 2 indicates high risk of poor outcome
    pub fn calculate_qsofa(vitals: &SepsisVitals) -> u8 {
        let mut score = 0;

        if vitals.respiratory_rate >= 22.0 {
            score += 1;
        }

        if vitals.mental_status_altered {
            score += 1;
        }

        if vitals.systolic_bp <= 100.0 {
            score += 1;
        }

        score
    }

    /// Calculate SIRS (Systemic Inflammatory Response Syndrome) score
    ///
    /// SIRS criteria:
    /// - Temperature > 38°C or < 36°C
    /// - Heart rate > 90 bpm
    /// - Respiratory rate > 20/min
    /// - WBC > 12,000 or < 4,000 or > 10% bands
    pub fn calculate_sirs(vitals: &SepsisVitals) -> u8 {
        let mut score = 0;

        // Temperature
        if vitals.temperature > 38.0 || vitals.temperature < 36.0 {
            score += 1;
        }

        // Heart rate
        if vitals.heart_rate > 90.0 {
            score += 1;
        }

        // Respiratory rate
        if vitals.respiratory_rate > 20.0 {
            score += 1;
        }

        // WBC count (if available)
        if let Some(wbc) = vitals.wbc_count {
            if wbc > 12.0 || wbc < 4.0 {
                score += 1;
            }
        }

        score
    }

    /// Comprehensive sepsis risk assessment
    pub fn assess_risk(vitals: &SepsisVitals) -> Result<SepsisRiskScore> {
        let qsofa_score = Self::calculate_qsofa(vitals);
        let sirs_score = Self::calculate_sirs(vitals);

        // Determine risk level
        let sepsis_risk = if qsofa_score >= 2 {
            SepsisRisk::High
        } else if sirs_score >= 2 {
            SepsisRisk::Moderate
        } else {
            SepsisRisk::Low
        };

        // Clinical recommendation
        let recommendation = match sepsis_risk {
            SepsisRisk::Low => "Continue monitoring. No immediate sepsis concern.".to_string(),
            SepsisRisk::Moderate => {
                "SIRS criteria met. Consider infection workup and close monitoring.".to_string()
            }
            SepsisRisk::High => {
                "HIGH RISK: qSOFA ≥ 2. Immediate sepsis protocol activation required. \
                 Consider ICU transfer, blood cultures, and empiric antibiotics."
                    .to_string()
            }
        };

        Ok(SepsisRiskScore {
            qsofa_score,
            sirs_score,
            sepsis_risk,
            recommendation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qsofa_high_risk() {
        let vitals = SepsisVitals {
            respiratory_rate: 24.0,
            systolic_bp: 95.0,
            mental_status_altered: true,
            temperature: 37.0,
            heart_rate: 80.0,
            wbc_count: None,
        };

        let score = SepsisRiskCalculator::calculate_qsofa(&vitals);
        assert_eq!(score, 3);
    }

    #[test]
    fn test_sepsis_risk_assessment() {
        let vitals = SepsisVitals {
            respiratory_rate: 24.0,
            systolic_bp: 95.0,
            mental_status_altered: true,
            temperature: 38.5,
            heart_rate: 105.0,
            wbc_count: Some(15.0),
        };

        let assessment = SepsisRiskCalculator::assess_risk(&vitals).unwrap();
        assert_eq!(assessment.sepsis_risk, SepsisRisk::High);
    }
}
