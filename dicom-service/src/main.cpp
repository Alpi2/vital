#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>
#include <memory>
#include <string>
#include <iostream>
#include <csignal>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "dicom_service_impl.h"
// #include "health_service.h"  # Disabled for now
#include "storage_manager.h"
#include "audit_logger.h"

using grpc::Server;
using grpc::ServerBuilder;
using vitalstream::dicom::DICOMServiceImpl;
// using vitalstream::dicom::HealthService;  # Disabled for now
using vitalstream::dicom::StorageManager;
using vitalstream::dicom::AuditLogger;

// Global server instance for graceful shutdown
std::unique_ptr<Server> server;
std::unique_ptr<DICOMServiceImpl> dicom_service;
// std::unique_ptr<HealthService> health_service;  # Disabled for now
std::shared_ptr<StorageManager> storage_manager;
std::shared_ptr<AuditLogger> audit_logger;

void signal_handler(int signal) {
    spdlog::info("🛑 Received signal {}, shutting down gracefully...", signal);
    
    if (server) {
        spdlog::info("🛑 Stopping gRPC server...");
        server->Shutdown();
    }
}

void run_server(const std::string& server_address, const std::string& storage_path) {
    try {
        spdlog::info("🏥 Starting VitalStream DICOM Service");
        spdlog::info("📍 Server address: {}", server_address);
        spdlog::info("📁 Storage path: {}", storage_path);
        
        // Initialize components
        spdlog::info("🔧 Initializing components...");
        
        // Initialize audit logger
        audit_logger = std::make_shared<AuditLogger>(storage_path + "/logs/dicom_audit.log");
        spdlog::info("✅ Audit logger initialized");
        
        // Initialize storage manager
        storage_manager = std::make_shared<StorageManager>(storage_path);
        spdlog::info("✅ Storage manager initialized");
        
        // Initialize health service - Disabled for now
        // health_service = std::make_unique<HealthService>(storage_manager, audit_logger);
        // spdlog::info("✅ Health service initialized");
        
        // Initialize DICOM service
        dicom_service = std::make_unique<DICOMServiceImpl>(storage_path, audit_logger);
        spdlog::info("✅ DICOM service initialized");
        
        // Build gRPC server
        ServerBuilder builder;
        
        // Add listening port
        grpc::SslServerCredentialsOptions ssl_opts;
        std::shared_ptr<grpc::ServerCredentials> creds;
        
        // For now, use insecure credentials (TODO: Add TLS support)
        creds = grpc::InsecureServerCredentials();
        
        builder.AddListeningPort(server_address, creds);
        
        // Register services
        builder.RegisterService(dicom_service.get());
        // builder.RegisterService(health_service.get()); // Disabled for now
        
        // Enable reflection (optional)
        // grpc::reflection::InitProtoReflectionServerBuilderPlugin(builder);
        
        // Set compression
        builder.SetDefaultCompressionAlgorithm(GRPC_COMPRESS_NONE);
        
        // Build and start server
        server = builder.BuildAndStart();
        spdlog::info("🚀 DICOM service listening on {}", server_address);
        
        // Log startup
        audit_logger->log_system_event("SERVICE_STARTUP", 
                                       "VitalStream DICOM Service started",
                                       "{\"address\":\"" + server_address + "\",\"storage_path\":\"" + storage_path + "\"}");
        
        // Wait for server shutdown
        server->Wait();
        
        spdlog::info("🛑 DICOM service stopped");
        
        // Log shutdown
        audit_logger->log_system_event("SERVICE_SHUTDOWN", 
                                       "VitalStream DICOM Service stopped",
                                       "{\"reason\":\"graceful_shutdown\"}");
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Server startup failed: {}", e.what());
        
        if (audit_logger) {
            audit_logger->log_error("SERVICE_STARTUP", e.what());
        }
        
        exit(1);
    }
}

int main(int argc, char** argv) {
    // Initialize logging
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^] [%l] %v");
    
    spdlog::info("🏥 VitalStream DICOM Service v1.0.0");
    
    // Parse command line arguments
    std::string server_address = "0.0.0.0:50051";
    std::string storage_path = "./dicom_storage";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--address" && i + 1 < argc) {
            server_address = argv[++i];
        } else if (arg == "--storage" && i + 1 < argc) {
            storage_path = argv[++i];
        } else if (arg == "--help") {
            std::cout << "VitalStream DICOM Service\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --address <addr>  Server address (default: 0.0.0.0:50051)\n";
            std::cout << "  --storage <path>  Storage directory (default: ./dicom_storage)\n";
            std::cout << "  --help           Show this help message\n\n";
            std::cout << "Examples:\n";
            std::cout << "  " << argv[0] << " --address 0.0.0.0:50051 --storage /data/dicom\n";
            std::cout << "  " << argv[0] << " --address localhost:50051\n";
            return 0;
        } else {
            spdlog::error("❌ Unknown argument: {}", arg);
            spdlog::error("Use --help for usage information");
            return 1;
        }
    }
    
    // Set up signal handlers for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    // Run the server
    run_server(server_address, storage_path);
    
    return 0;
}
