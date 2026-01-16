//! Cardiac Arrest Risk Assessment
//!
//! Predictive algorithms for cardiac arrest risk.

use crate::{ArrhythmiaType, InferenceError, Result};

/// Cardiac arrest risk assessment
#[derive(Debug, Clone)]
pub struct CardiacArrestRisk {
    pub risk_score: f32,        // 0.0-1.0
    pub risk_level: RiskLevel,
    pub contributing_factors: Vec<String>,
    pub time_to_event_hours: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Moderate,
    High,
    Critical,
}

/// Factors contributing to cardiac arrest risk
#[derive(Debug, Clone)]
pub struct RiskFactors {
    pub arrhythmia: Option<ArrhythmiaType>,
    pub heart_rate: f32,
    pub systolic_bp: f32,
    pub oxygen_saturation: f32,
    pub respiratory_rate: f32,
    pub qt_interval: Option<f32>,  // QTc in ms
    pub previous_cardiac_event: bool,
    pub age: u8,
}

pub struct CardiacArrestPredictor;

impl CardiacArrestPredictor {
    /// Assess cardiac arrest risk
    pub fn assess_risk(factors: &RiskFactors) -> Result<CardiacArrestRisk> {
        let mut risk_score = 0.0;
        let mut contributing_factors = Vec::new();

        // Arrhythmia contribution
        if let Some(arrhythmia) = factors.arrhythmia {
            let arrhythmia_risk = match arrhythmia {
                ArrhythmiaType::VentricularFibrillation => {
                    contributing_factors.push("Ventricular fibrillation detected".to_string());
                    0.9
                }
                ArrhythmiaType::VentricularTachycardia => {
                    contributing_factors.push("Ventricular tachycardia detected".to_string());
                    0.7
                }
                ArrhythmiaType::STEMI => {
                    contributing_factors.push("STEMI detected".to_string());
                    0.6
                }
                ArrhythmiaType::Bradycardia if factors.heart_rate < 40.0 => {
                    contributing_factors.push("Severe bradycardia".to_string());
                    0.5
                }
                _ => 0.1,
            };
            risk_score += arrhythmia_risk;
        }

        // Vital signs contribution
        if factors.systolic_bp < 90.0 {
            contributing_factors.push("Hypotension (SBP < 90 mmHg)".to_string());
            risk_score += 0.3;
        }

        if factors.oxygen_saturation < 90.0 {
            contributing_factors.push("Severe hypoxemia (SpO2 < 90%)".to_string());
            risk_score += 0.3;
        }

        if factors.respiratory_rate > 30.0 || factors.respiratory_rate < 8.0 {
            contributing_factors.push("Abnormal respiratory rate".to_string());
            risk_score += 0.2;
        }

        // QTc prolongation
        if let Some(qtc) = factors.qt_interval {
            if qtc > 500.0 {
                contributing_factors.push("Prolonged QTc interval (> 500ms)".to_string());
                risk_score += 0.4;
            }
        }

        // Previous cardiac event
        if factors.previous_cardiac_event {
            contributing_factors.push("History of cardiac events".to_string());
            risk_score += 0.2;
        }

        // Age factor
        if factors.age > 75 {
            risk_score += 0.1;
        }

        // Normalize risk score
        risk_score = risk_score.min(1.0);

        // Determine risk level
        let risk_level = match risk_score {
            x if x < 0.3 => RiskLevel::Low,
            x if x < 0.5 => RiskLevel::Moderate,
            x if x < 0.7 => RiskLevel::High,
            _ => RiskLevel::Critical,
        };

        // Estimate time to event (very simplified)
        let time_to_event_hours = if risk_score > 0.7 {
            Some(1.0)  // < 1 hour for critical
        } else if risk_score > 0.5 {
            Some(6.0)  // < 6 hours for high
        } else {
            None
        };

        Ok(CardiacArrestRisk {
            risk_score,
            risk_level,
            contributing_factors,
            time_to_event_hours,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_critical_risk() {
        let factors = RiskFactors {
            arrhythmia: Some(ArrhythmiaType::VentricularFibrillation),
            heart_rate: 150.0,
            systolic_bp: 80.0,
            oxygen_saturation: 85.0,
            respiratory_rate: 35.0,
            qt_interval: Some(520.0),
            previous_cardiac_event: true,
            age: 80,
        };

        let risk = CardiacArrestPredictor::assess_risk(&factors).unwrap();
        assert_eq!(risk.risk_level, RiskLevel::Critical);
        assert!(risk.risk_score > 0.7);
    }
}
