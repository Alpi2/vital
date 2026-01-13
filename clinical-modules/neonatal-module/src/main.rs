use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeonatalPatient {
    pub id: String,
    pub gestational_age_weeks: u8,
    pub birth_weight_grams: u32,
    pub apgar_1min: u8,
    pub apgar_5min: u8,
}

async fn health() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({"status": "ok", "module": "neonatal"}))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().route("/health", web::get().to(health)))
        .bind(("127.0.0.1", 8087))?
        .run()
        .await
}
