use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ORPatient {
    pub id: String,
    pub name: String,
    pub procedure: String,
    pub anesthesia_type: String,
}

async fn health_check() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({"status": "ok", "module": "or"}))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().route("/health", web::get().to(health_check)))
        .bind(("127.0.0.1", 8084))?
        .run()
        .await
}
