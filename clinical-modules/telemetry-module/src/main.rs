use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryPatient {
    pub id: String,
    pub name: String,
    pub room: String,
    pub rhythm: String,
    pub heart_rate: u32,
}

async fn health_check() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({"status": "ok", "module": "telemetry"}))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().route("/health", web::get().to(health_check)))
        .bind(("127.0.0.1", 8083))?
        .run()
        .await
}
