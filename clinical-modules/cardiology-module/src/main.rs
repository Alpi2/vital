use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardiologyPatient {
    pub id: String,
    pub name: String,
    pub diagnosis: String,
    pub ejection_fraction: Option<f64>,
    pub troponin: Option<f64>,
    pub bnp: Option<f64>,
    pub ecg_findings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTest {
    pub patient_id: String,
    pub protocol: String,
    pub duration: u32,
    pub max_heart_rate: u32,
    pub st_changes: bool,
    pub symptoms: Vec<String>,
}

async fn health_check() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "ok",
        "module": "cardiology"
    }))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    
    HttpServer::new(|| {
        App::new()
            .route("/health", web::get().to(health_check))
    })
    .bind(("127.0.0.1", 8082))?
    .run()
    .await
}
