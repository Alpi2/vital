use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};

// Stress Test Support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTest {
    pub patient_id: String,
    pub protocol: StressTestProtocol,
    pub duration_minutes: u32,
    pub max_heart_rate: u32,
    pub target_heart_rate: u32,
    pub st_changes: bool,
    pub symptoms: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StressTestProtocol {
    Bruce,
    ModifiedBruce,
    Naughton,
    Balke,
}

// Holter Monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolterRecording {
    pub patient_id: String,
    pub duration_hours: u32,
    pub total_beats: u64,
    pub arrhythmia_events: Vec<ArrhythmiaEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrhythmiaEvent {
    pub timestamp: String,
    pub event_type: String,
    pub duration_seconds: u32,
}

// Pacemaker Check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacemakerCheck {
    pub patient_id: String,
    pub pacing_detected: bool,
    pub capture_confirmed: bool,
    pub sensing_threshold: f64,
}

// Defibrillation Support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefibrillationEvent {
    pub patient_id: String,
    pub energy_joules: u32,
    pub rhythm_before: String,
    pub rhythm_after: String,
    pub success: bool,
}

async fn start_stress_test(test: web::Json<StressTest>) -> impl Responder {
    HttpResponse::Ok().json(test.into_inner())
}

async fn record_holter(recording: web::Json<HolterRecording>) -> impl Responder {
    HttpResponse::Ok().json(recording.into_inner())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/stress-test", web::post().to(start_stress_test))
            .route("/holter", web::post().to(record_holter))
    })
    .bind(("127.0.0.1", 8089))?
    .run()
    .await
}
