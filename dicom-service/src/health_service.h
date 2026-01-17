#pragma once

#include <grpcpp/grpcpp.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>
#include "common/v1/common.grpc.pb.h"
#include "storage_manager.h"
#include "audit_logger.h"

namespace vitalstream {
namespace dicom {

/**
 * Health check result structure
 */
struct HealthCheckResult {
    common::v1::HealthCheckResponse::ServingStatus status;
    size_t uptime_seconds;
    size_t memory_usage_mb;
    double disk_usage_percent;
    std::string storage_status;
    
    HealthCheckResult() : status(common::v1::HealthCheckResponse::SERVING_STATUS_SERVING),
                         uptime_seconds(0), memory_usage_mb(0), 
                         disk_usage_percent(0.0), storage_status("unknown") {}
};

/**
 * Health status tracking
 */
struct HealthStatus {
    common::v1::HealthCheckResponse::ServingStatus status;
    std::chrono::system_clock::time_point last_check;
    std::chrono::system_clock::time_point uptime_start;
    size_t total_checks;
    size_t failed_checks;
    
    HealthStatus() : status(common::v1::HealthCheckResponse::SERVING_STATUS_SERVING),
                     total_checks(0), failed_checks(0) {
        uptime_start = std::chrono::system_clock::now();
        last_check = uptime_start;
    }
};

/**
 * Health Service implementation
 * 
 * Provides comprehensive health monitoring for the DICOM service
 * including storage, memory, disk, and operational health checks.
 */
class HealthService {
public:
    explicit HealthService(std::shared_ptr<StorageManager> storage_manager = nullptr,
                          std::shared_ptr<AuditLogger> audit_logger = nullptr);
    ~HealthService();
    
    /**
     * gRPC health check implementation
     * 
     * @param context gRPC server context
     * @param request Health check request
     * @param response Health check response
     * @return gRPC status
     */
    grpc::Status CheckHealth(grpc::ServerContext* context,
                             const common::v1::HealthCheckRequest* request,
                             common::v1::HealthCheckResponse* response);
    
    /**
     * Get current health status
     * 
     * @return Current health status
     */
    HealthStatus get_health_status();
    
    /**
     * Get detailed health information
     * 
     * @return Map with detailed health metrics
     */
    std::map<std::string, std::string> get_detailed_health();
    
    /**
     * Manually set health status
     * 
     * @param status New health status
     */
    void set_health_status(common::v1::HealthCheckResponse::ServingStatus status);

private:
    std::shared_ptr<StorageManager> storage_manager_;
    std::shared_ptr<AuditLogger> audit_logger_;
    HealthStatus health_status_;
    mutable std::mutex status_mutex_;
    
    // Background monitoring
    std::atomic<bool> monitoring_active_;
    std::thread monitoring_thread_;
    
    /**
     * Perform comprehensive health check
     * 
     * @return Health check result
     */
    HealthCheckResult perform_health_check();
    
    /**
     * Background monitoring loop
     */
    void monitoring_loop();
    
    /**
     * Get current memory usage in MB
     * 
     * @return Memory usage in MB
     */
    size_t get_memory_usage_mb();
    
    /**
     * Get disk usage percentage
     * 
     * @return Disk usage percentage (0-100)
     */
    double get_disk_usage_percent();
    
    /**
     * Format timestamp for logging
     * 
     * @param timestamp Time point to format
     * @return Formatted timestamp string
     */
    std::string format_timestamp(const std::chrono::system_clock::time_point& timestamp);
    
    /**
     * Generate health details JSON
     * 
     * @param result Health check result
     * @return JSON string with health details
     */
    std::string generate_health_details_json(const HealthCheckResult& result);
    
    /**
     * Generate health statistics JSON
     * 
     * @return JSON string with health statistics
     */
    std::string generate_health_stats_json();
};

} // namespace dicom
} // namespace vitalstream
