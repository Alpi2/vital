use actix_web::{web, App, HttpResponse, HttpServer, Responder};

async fn health() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({"status": "ok", "module": "stepdown"}))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().route("/health", web::get().to(health)))
        .bind(("127.0.0.1", 8088))?
        .run()
        .await
}
