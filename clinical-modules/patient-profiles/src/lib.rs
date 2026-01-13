use serde::{Deserialize, Serialize};

// Pediatric Profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PediatricProfile {
    pub age_months: u32,
    pub weight_kg: f64,
    pub height_cm: f64,
    pub vital_ranges: PediatricVitalRanges,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PediatricVitalRanges {
    pub hr_min: u32,
    pub hr_max: u32,
    pub rr_min: u32,
    pub rr_max: u32,
    pub sbp_min: u32,
    pub sbp_max: u32,
}

impl PediatricProfile {
    pub fn new(age_months: u32, weight_kg: f64, height_cm: f64) -> Self {
        let vital_ranges = Self::calculate_ranges(age_months);
        Self { age_months, weight_kg, height_cm, vital_ranges }
    }
    
    fn calculate_ranges(age_months: u32) -> PediatricVitalRanges {
        match age_months {
            0..=3 => PediatricVitalRanges { hr_min: 100, hr_max: 160, rr_min: 30, rr_max: 60, sbp_min: 60, sbp_max: 90 },
            4..=12 => PediatricVitalRanges { hr_min: 90, hr_max: 150, rr_min: 25, rr_max: 50, sbp_min: 70, sbp_max: 100 },
            _ => PediatricVitalRanges { hr_min: 70, hr_max: 120, rr_min: 20, rr_max: 30, sbp_min: 80, sbp_max: 110 },
        }
    }
}

// Geriatric Profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeriatricProfile {
    pub age_years: u32,
    pub frailty_score: u8,
    pub comorbidities: Vec<String>,
}

// Obese Profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObeseProfile {
    pub bmi: f64,
    pub adjusted_drug_dosing: bool,
}

// Pregnant Profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PregnantProfile {
    pub trimester: u8,
    pub gestational_weeks: u8,
    pub fetal_monitoring: bool,
}

// Pacemaker/ICD Profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacemakerProfile {
    pub device_type: String,
    pub manufacturer: String,
    pub magnet_mode_enabled: bool,
}

// Post-operative Profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostOperativeProfile {
    pub procedure: String,
    pub hours_post_op: u32,
    pub pacu_score: u8,
}

// Cardiac Surgery Profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardiacSurgeryProfile {
    pub procedure_type: String,
    pub post_cpb: bool,
    pub chest_tubes: u8,
}
