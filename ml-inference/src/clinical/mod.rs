//! Clinical Decision Support Algorithms
//!
//! Evidence-based clinical scoring systems and risk assessment.

pub mod ews;
pub mod sepsis;
pub mod cardiac;

pub use ews::{EarlyWarningScore, NEWS2Score};
pub use sepsis::SepsisRiskScore;
pub use cardiac::CardiacArrestRisk;
