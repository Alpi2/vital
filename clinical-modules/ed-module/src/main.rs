use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EDPatient {
    pub id: String,
    pub name: String,
    pub chief_complaint: String,
    pub esi_level: u8, // Emergency Severity Index 1-5
    pub arrival_time: String,
    pub triage_vitals: TriageVitals,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriageVitals {
    pub heart_rate: u32,
    pub blood_pressure: String,
    pub respiratory_rate: u32,
    pub temperature: f64,
    pub spo2: u32,
    pub pain_score: u8,
}

async fn triage_patient(patient: web::Json<EDPatient>) -> impl Responder {
    HttpResponse::Ok().json(patient.into_inner())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/triage", web::post().to(triage_patient))
    })
    .bind(("127.0.0.1", 8085))?
    .run()
    .await
}
