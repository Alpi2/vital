use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ICUPatient {
    pub id: String,
    pub name: String,
    pub bed: String,
    pub apache_score: Option<f64>,
    pub sofa_score: Option<f64>,
    pub ventilator_settings: Option<VentilatorSettings>,
    pub hemodynamic_status: Option<HemodynamicStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VentilatorSettings {
    pub mode: String,
    pub tidal_volume: f64,
    pub respiratory_rate: u32,
    pub peep: f64,
    pub fio2: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HemodynamicStatus {
    pub map: f64,
    pub cvp: f64,
    pub cardiac_output: f64,
    pub svr: f64,
}

struct AppState {
    patients: Mutex<Vec<ICUPatient>>,
}

async fn get_patients(data: web::Data<AppState>) -> impl Responder {
    let patients = data.patients.lock().unwrap();
    HttpResponse::Ok().json(&*patients)
}

async fn add_patient(patient: web::Json<ICUPatient>, data: web::Data<AppState>) -> impl Responder {
    let mut patients = data.patients.lock().unwrap();
    patients.push(patient.into_inner());
    HttpResponse::Created().finish()
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    
    let app_state = web::Data::new(AppState {
        patients: Mutex::new(Vec::new()),
    });
    
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .route("/patients", web::get().to(get_patients))
            .route("/patients", web::post().to(add_patient))
    })
    .bind(("127.0.0.1", 8081))?
    .run()
    .await
}
