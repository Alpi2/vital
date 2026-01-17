#include "health_service.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <thread>
#include <filesystem>

namespace vitalstream {
namespace dicom {

HealthService::HealthService(std::shared_ptr<StorageManager> storage_manager,
                             std::shared_ptr<AuditLogger> audit_logger)
    : storage_manager_(storage_manager), audit_logger_(audit_logger) {
    spdlog::info("🏥 Initializing Health Service");
    
    // Initialize health status
    health_status_.status = common::v1::HealthCheckResponse::SERVING;
    health_status_.last_check = std::chrono::system_clock::now();
    health_status_.uptime_start = std::chrono::system_clock::now();
    health_status_.total_checks = 0;
    health_status_.failed_checks = 0;
    
    // Start background health monitoring
    monitoring_active_ = true;
    monitoring_thread_ = std::thread(&HealthService::monitoring_loop, this);
}

HealthService::~HealthService() {
    spdlog::info("🏥 Health Service shutting down");
    
    // Stop monitoring
    monitoring_active_ = false;
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
    
    // Log final health statistics
    audit_logger_->log_system_event("HEALTH_SERVICE_SHUTDOWN", 
                                   "Health service monitoring stopped",
                                   generate_health_stats_json());
}

grpc::Status HealthService::CheckHealth(
    grpc::ServerContext* context,
    const common::v1::HealthCheckRequest* request,
    common::v1::HealthCheckResponse* response
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    spdlog::debug("🏥 Health check requested");
    
    try {
        // Perform comprehensive health check
        HealthCheckResult result = perform_health_check();
        
        // Set response
        response->set_status(result.status);
        
        // Add detailed health information
        auto* details = response->mutable_details();
        details->insert({"service", "dicom-service"});
        details->insert({"version", "1.0.0"});
        details->insert({"uptime_seconds", std::to_string(result.uptime_seconds)});
        details->insert({"total_checks", std::to_string(health_status_.total_checks)});
        details->insert({"failed_checks", std::to_string(health_status_.failed_checks)});
        details->insert({"storage_status", result.storage_status});
        details->insert({"memory_usage_mb", std::to_string(result.memory_usage_mb)});
        details->insert({"disk_usage_percent", std::to_string(result.disk_usage_percent)});
        
        // Calculate check duration
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        ).count();
        
        details->insert({"check_duration_ms", std::to_string(duration)});
        
        // Update health status
        {
            std::lock_guard<std::mutex> lock(status_mutex_);
            health_status_.last_check = std::chrono::system_clock::now();
            health_status_.total_checks++;
            health_status_.status = result.status;
            
            if (result.status != common::v1::HealthCheckResponse::SERVING) {
                health_status_.failed_checks++;
            }
        }
        
        // Log health check
        std::string status_str = (result.status == common::v1::HealthCheckResponse::SERVING) ? "HEALTHY" : "UNHEALTHY";
        spdlog::info("🏥 Health check: {} ({}ms)", status_str, duration);
        
        // Audit log health checks (only failures to reduce noise)
        if (result.status != common::v1::HealthCheckResponse::SERVING) {
            audit_logger_->log_system_event("HEALTH_CHECK_FAILED", 
                                           "Service health check failed",
                                           generate_health_details_json(result));
        }
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Health check exception: {}", e.what());
        
        response->set_status(common::v1::HealthCheckResponse::NOT_SERVING);
        response->mutable_details()->insert({"error", e.what()});
        
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
}

HealthCheckResult HealthService::perform_health_check() {
    HealthCheckResult result;
    
    auto now = std::chrono::system_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - health_status_.uptime_start);
    result.uptime_seconds = uptime.count();
    
    // Default to serving
    result.status = common::v1::HealthCheckResponse::SERVING;
    
    // Check storage manager
    if (storage_manager_) {
        auto storage_stats = storage_manager_->get_statistics();
        result.storage_status = "healthy";
        
        // Check storage efficiency
        double efficiency = storage_manager_->get_storage_efficiency();
        if (efficiency < 0.5) {
            result.status = common::v1::HealthCheckResponse::NOT_SERVING;
            result.storage_status = "inefficient";
        }
        
        // Verify storage integrity
        if (!storage_manager_->verify_storage_integrity()) {
            result.status = common::v1::HealthCheckResponse::NOT_SERVING;
            result.storage_status = "corrupted";
        }
    } else {
        result.status = common::v1::HealthCheckResponse::NOT_SERVING;
        result.storage_status = "unavailable";
    }
    
    // Check memory usage
    result.memory_usage_mb = get_memory_usage_mb();
    if (result.memory_usage_mb > 1024) { // More than 1GB
        result.status = common::v1::HealthCheckResponse::NOT_SERVING;
    }
    
    // Check disk usage
    result.disk_usage_percent = get_disk_usage_percent();
    if (result.disk_usage_percent > 90) {
        result.status = common::v1::HealthCheckResponse::NOT_SERVING;
    }
    
    return result;
}

HealthStatus HealthService::get_health_status() {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return health_status_;
}

std::map<std::string, std::string> HealthService::get_detailed_health() {
    HealthCheckResult result = perform_health_check();
    
    std::map<std::string, std::string> details;
    details["service"] = "dicom-service";
    details["version"] = "1.0.0";
    details["status"] = (result.status == common::v1::HealthCheckResponse::SERVING) ? "healthy" : "unhealthy";
    details["uptime_seconds"] = std::to_string(result.uptime_seconds);
    details["memory_usage_mb"] = std::to_string(result.memory_usage_mb);
    details["disk_usage_percent"] = std::to_string(result.disk_usage_percent);
    details["storage_status"] = result.storage_status;
    details["total_checks"] = std::to_string(health_status_.total_checks);
    details["failed_checks"] = std::to_string(health_status_.failed_checks);
    details["last_check"] = format_timestamp(health_status_.last_check);
    
    return details;
}

void HealthService::set_health_status(common::v1::HealthCheckResponse::ServingStatus status) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    health_status_.status = status;
    health_status_.last_check = std::chrono::system_clock::now();
    
    std::string status_str = (status == common::v1::HealthCheckResponse::SERVING) ? "SERVING" : "NOT_SERVING";
    spdlog::info("🏥 Health status manually set to: {}", status_str);
    
    audit_logger_->log_system_event("HEALTH_STATUS_MANUAL_SET", 
                                   "Health status manually updated",
                                   "{\"status\":\"" + status_str + "\"}");
}

// Private methods

void HealthService::monitoring_loop() {
    spdlog::info("🏥 Starting health monitoring loop");
    
    while (monitoring_active_) {
        try {
            // Perform health check
            HealthCheckResult result = perform_health_check();
            
            // Log if status changed
            common::v1::HealthCheckResponse::ServingStatus old_status;
            {
                std::lock_guard<std::mutex> lock(status_mutex_);
                old_status = health_status_.status;
                health_status_.status = result.status;
            }
            
            if (old_status != result.status) {
                std::string old_status_str = (old_status == common::v1::HealthCheckResponse::SERVING) ? "HEALTHY" : "UNHEALTHY";
                std::string new_status_str = (result.status == common::v1::HealthCheckResponse::SERVING) ? "HEALTHY" : "UNHEALTHY";
                
                spdlog::warn("🏥 Health status changed: {} -> {}", old_status_str, new_status_str);
                
                audit_logger_->log_system_event("HEALTH_STATUS_CHANGED", 
                                               "Service health status changed",
                                               "{\"old_status\":\"" + old_status_str + "\",\"new_status\":\"" + new_status_str + "\"}");
            }
            
            // Sleep for monitoring interval
            std::this_thread::sleep_for(std::chrono::seconds(30));
            
        } catch (const std::exception& e) {
            spdlog::error("💥 Health monitoring loop error: {}", e.what());
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
    }
    
    spdlog::info("🏥 Health monitoring loop stopped");
}

size_t HealthService::get_memory_usage_mb() {
    // Get memory usage (platform-specific)
    size_t memory_kb = 0;
    
#ifdef __linux__
    std::ifstream status_file("/proc/self/status");
    if (status_file.is_open()) {
        std::string line;
        while (std::getline(status_file, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::istringstream iss(line);
                std::string label, value, unit;
                iss >> label >> value >> unit;
                memory_kb = std::stoul(value);
                break;
            }
        }
        status_file.close();
    }
#elif defined(__APPLE__)
    // macOS: use vm_statistics
    struct task_basic_info t_info;
    mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;
    
    if (task_info(mach_task_self(), TASK_BASIC_INFO, 
                   (task_info_t)&t_info, &t_info_count) == KERN_SUCCESS) {
        memory_kb = t_info.resident_size / 1024;
    }
#else
    // Windows or other platforms
    memory_kb = 100000; // Placeholder
#endif
    
    return memory_kb / 1024; // Convert to MB
}

double HealthService::get_disk_usage_percent() {
    try {
        std::filesystem::space_info space = std::filesystem::space(".");
        
        if (space.capacity == 0) {
            return 0.0;
        }
        
        double used_percent = (static_cast<double>(space.capacity - space.available) / space.capacity) * 100.0;
        return used_percent;
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Error getting disk usage: {}", e.what());
        return 0.0;
    }
}

std::string HealthService::format_timestamp(const std::chrono::system_clock::time_point& timestamp) {
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

std::string HealthService::generate_health_details_json(const HealthCheckResult& result) {
    std::stringstream ss;
    ss << "{"
       << "\"uptime_seconds\":" << result.uptime_seconds << ","
       << "\"memory_usage_mb\":" << result.memory_usage_mb << ","
       << "\"disk_usage_percent\":" << result.disk_usage_percent << ","
       << "\"storage_status\":\"" << result.storage_status << "\""
       << "}";
    return ss.str();
}

std::string HealthService::generate_health_stats_json() {
    std::lock_guard<std::mutex> lock(status_mutex_);
    
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now() - health_status_.uptime_start
    );
    
    std::stringstream ss;
    ss << "{"
       << "\"total_operations\":" << health_status_.total_checks << ","
       << "\"failed_operations\":" << health_status_.failed_checks << ","
       << "\"uptime_seconds\":" << uptime.count() << ","
       << "\"success_rate\":" << (health_status_.total_checks > 0 ? 
           (static_cast<double>(health_status_.total_checks - health_status_.failed_checks) / health_status_.total_checks * 100.0) : 0.0)
       << "}";
    return ss.str();
}

} // namespace dicom
} // namespace vitalstream
