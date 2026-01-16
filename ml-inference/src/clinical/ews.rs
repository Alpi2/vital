//! Early Warning Score (EWS) Calculations
//!
//! Implements NEWS2 (National Early Warning Score 2) and MEWS.

use crate::{InferenceError, Result};

/// Vital signs for EWS calculation
#[derive(Debug, Clone)]
pub struct VitalSigns {
    pub respiratory_rate: f32,      // breaths/min
    pub oxygen_saturation: f32,     // %
    pub systolic_bp: f32,           // mmHg
    pub heart_rate: f32,            // bpm
    pub temperature: f32,           // °C
    pub consciousness_level: ConsciousnessLevel,
    pub supplemental_oxygen: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsciousnessLevel {
    Alert,
    VoiceResponse,
    PainResponse,
    Unresponsive,
}

/// Early Warning Score result
#[derive(Debug, Clone)]
pub struct EarlyWarningScore {
    pub total_score: u8,
    pub risk_level: RiskLevel,
    pub recommended_action: String,
    pub component_scores: ComponentScores,
}

#[derive(Debug, Clone)]
pub struct ComponentScores {
    pub respiratory_rate: u8,
    pub oxygen_saturation: u8,
    pub systolic_bp: u8,
    pub heart_rate: u8,
    pub temperature: u8,
    pub consciousness: u8,
    pub supplemental_oxygen: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Low,        // 0-4
    Medium,     // 5-6
    High,       // 7+
}

/// NEWS2 (National Early Warning Score 2) calculator
pub struct NEWS2Score;

impl NEWS2Score {
    /// Calculate NEWS2 score from vital signs
    ///
    /// # Arguments
    ///
    /// * `vitals` - Patient vital signs
    ///
    /// # Returns
    ///
    /// Early warning score with risk assessment
    pub fn calculate(vitals: &VitalSigns) -> Result<EarlyWarningScore> {
        let mut components = ComponentScores {
            respiratory_rate: 0,
            oxygen_saturation: 0,
            systolic_bp: 0,
            heart_rate: 0,
            temperature: 0,
            consciousness: 0,
            supplemental_oxygen: 0,
        };

        // Respiratory rate scoring
        components.respiratory_rate = match vitals.respiratory_rate as u8 {
            0..=8 => 3,
            9..=11 => 1,
            12..=20 => 0,
            21..=24 => 2,
            _ => 3,
        };

        // SpO2 scoring (Scale 1 - for most patients)
        components.oxygen_saturation = match vitals.oxygen_saturation as u8 {
            0..=91 => 3,
            92..=93 => 2,
            94..=95 => 1,
            _ => 0,
        };

        // Supplemental oxygen
        components.supplemental_oxygen = if vitals.supplemental_oxygen { 2 } else { 0 };

        // Systolic BP scoring
        components.systolic_bp = match vitals.systolic_bp as u8 {
            0..=90 => 3,
            91..=100 => 2,
            101..=110 => 1,
            111..=219 => 0,
            _ => 3,
        };

        // Heart rate scoring
        components.heart_rate = match vitals.heart_rate as u8 {
            0..=40 => 3,
            41..=50 => 1,
            51..=90 => 0,
            91..=110 => 1,
            111..=130 => 2,
            _ => 3,
        };

        // Temperature scoring
        let temp_score = if vitals.temperature <= 35.0 {
            3
        } else if vitals.temperature <= 36.0 {
            1
        } else if vitals.temperature <= 38.0 {
            0
        } else if vitals.temperature <= 39.0 {
            1
        } else {
            2
        };
        components.temperature = temp_score;

        // Consciousness level (AVPU)
        components.consciousness = match vitals.consciousness_level {
            ConsciousnessLevel::Alert => 0,
            _ => 3,
        };

        // Calculate total score
        let total_score = components.respiratory_rate
            + components.oxygen_saturation
            + components.supplemental_oxygen
            + components.systolic_bp
            + components.heart_rate
            + components.temperature
            + components.consciousness;

        // Determine risk level
        let risk_level = match total_score {
            0..=4 => RiskLevel::Low,
            5..=6 => RiskLevel::Medium,
            _ => RiskLevel::High,
        };

        // Recommended action
        let recommended_action = match risk_level {
            RiskLevel::Low => "Continue routine monitoring".to_string(),
            RiskLevel::Medium => {
                "Increase monitoring frequency. Inform registered nurse.".to_string()
            }
            RiskLevel::High => {
                "URGENT: Immediate clinical review required. Consider ICU transfer.".to_string()
            }
        };

        Ok(EarlyWarningScore {
            total_score,
            risk_level,
            recommended_action,
            component_scores: components,
        })
    }
}

/// MEWS (Modified Early Warning Score) calculator
pub struct MEWSScore;

impl MEWSScore {
    pub fn calculate(vitals: &VitalSigns) -> Result<EarlyWarningScore> {
        // Similar to NEWS2 but with different scoring criteria
        // TODO: Implement MEWS-specific scoring
        NEWS2Score::calculate(vitals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_news2_normal_vitals() {
        let vitals = VitalSigns {
            respiratory_rate: 16.0,
            oxygen_saturation: 98.0,
            systolic_bp: 120.0,
            heart_rate: 75.0,
            temperature: 37.0,
            consciousness_level: ConsciousnessLevel::Alert,
            supplemental_oxygen: false,
        };

        let score = NEWS2Score::calculate(&vitals).unwrap();
        assert_eq!(score.risk_level, RiskLevel::Low);
        assert_eq!(score.total_score, 0);
    }

    #[test]
    fn test_news2_critical_vitals() {
        let vitals = VitalSigns {
            respiratory_rate: 28.0,
            oxygen_saturation: 88.0,
            systolic_bp: 85.0,
            heart_rate: 135.0,
            temperature: 35.0,
            consciousness_level: ConsciousnessLevel::PainResponse,
            supplemental_oxygen: true,
        };

        let score = NEWS2Score::calculate(&vitals).unwrap();
        assert_eq!(score.risk_level, RiskLevel::High);
        assert!(score.total_score >= 7);
    }
}
