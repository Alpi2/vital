use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PediatricPatient {
    pub id: String,
    pub name: String,
    pub age_months: u32,
    pub weight_kg: f64,
    pub height_cm: f64,
    pub growth_percentile: GrowthPercentile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthPercentile {
    pub weight_percentile: f64,
    pub height_percentile: f64,
    pub bmi_percentile: f64,
}

async fn calculate_growth(patient: web::Json<PediatricPatient>) -> impl Responder {
    HttpResponse::Ok().json(patient.into_inner())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/growth", web::post().to(calculate_growth))
    })
    .bind(("127.0.0.1", 8086))?
    .run()
    .await
}
