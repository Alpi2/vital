#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <mutex>
#include <filesystem>

namespace vitalstream {
namespace dicom {

/**
 * Audit event structure
 */
struct AuditEvent {
    std::string timestamp;
    std::string event_type;
    std::string user_id;
    std::string patient_id;
    std::string dicom_id;
    std::string description;
    std::string details; // JSON string with additional details
};

/**
 * Audit statistics structure
 */
struct AuditStats {
    size_t total_operations;
    size_t dicom_stored;
    size_t dicom_retrieved;
    size_t dicom_deleted;
    size_t waveform_extracted;
    size_t access_denied;
    size_t errors;
    std::chrono::system_clock::time_point start_time;
    
    AuditStats() : total_operations(0), dicom_stored(0), dicom_retrieved(0),
                  dicom_deleted(0), waveform_extracted(0), access_denied(0), errors(0) {
        start_time = std::chrono::system_clock::now();
    }
};

/**
 * HIPAA-compliant audit logger for DICOM operations
 * 
 * Provides comprehensive logging of all DICOM operations with
 * proper security, integrity, and compliance features.
 */
class AuditLogger {
public:
    explicit AuditLogger(const std::string& log_file_path = "logs/dicom_audit.log");
    ~AuditLogger();
    
    /**
     * Log DICOM file storage
     * 
     * @param dicom_id Unique DICOM identifier
     * @param patient_id Patient identifier
     * @param modality DICOM modality
     */
    void log_dicom_stored(const std::string& dicom_id, 
                          const std::string& patient_id, 
                          const std::string& modality);
    
    /**
     * Log DICOM file retrieval
     * 
     * @param dicom_id Unique DICOM identifier
     * @param patient_id Patient identifier
     */
    void log_dicom_retrieved(const std::string& dicom_id, 
                            const std::string& patient_id);
    
    /**
     * Log DICOM file deletion
     * 
     * @param dicom_id Unique DICOM identifier
     * @param permanent Whether deletion was permanent
     */
    void log_dicom_deleted(const std::string& dicom_id, bool permanent);
    
    /**
     * Log waveform extraction
     * 
     * @param dicom_id Unique DICOM identifier
     * @param patient_id Patient identifier
     * @param num_channels Number of channels extracted
     * @param num_samples Number of samples extracted
     */
    void log_waveform_extracted(const std::string& dicom_id, 
                               const std::string& patient_id,
                               int num_channels,
                               int num_samples);
    
    /**
     * Log access denied event
     * 
     * @param user_id User identifier
     * @param resource Resource being accessed
     * @param reason Reason for denial
     */
    void log_access_denied(const std::string& user_id, 
                           const std::string& resource,
                           const std::string& reason);
    
    /**
     * Log error event
     * 
     * @param operation Operation that failed
     * @param error_message Error description
     * @param context Additional context
     */
    void log_error(const std::string& operation, 
                   const std::string& error_message,
                   const std::string& context = "");
    
    /**
     * Log system event
     * 
     * @param event_type Type of system event
     * @param description Event description
     * @param details Additional details (JSON)
     */
    void log_system_event(const std::string& event_type, 
                          const std::string& description,
                          const std::string& details = "");
    
    /**
     * Get audit statistics
     * 
     * @return Current audit statistics
     */
    AuditStats get_statistics();
    
    /**
     * Query audit events with filters
     * 
     * @param event_type Optional event type filter
     * @param user_id Optional user ID filter
     * @param from_time Optional start time filter
     * @param to_time Optional end time filter
     * @param limit Maximum number of events to return
     * @return List of matching audit events
     */
    std::vector<AuditEvent> query_events(const std::string& event_type = "",
                                        const std::string& user_id = "",
                                        const std::chrono::system_clock::time_point& from_time = std::chrono::system_clock::time_point{},
                                        const std::chrono::system_clock::time_point& to_time = std::chrono::system_clock::time_point{},
                                        int limit = 1000);

private:
    std::string log_file_path_;
    AuditStats stats_;
    mutable std::mutex log_mutex_;
    mutable std::mutex stats_mutex_;
    
    /**
     * Get current timestamp in ISO 8601 format
     * 
     * @return Timestamp string
     */
    std::string get_timestamp();
    
    /**
     * Generate DICOM operation details JSON
     * 
     * @param dicom_id DICOM identifier
     * @param patient_id Patient identifier
     * @param modality DICOM modality
     * @param user_id User identifier
     * @return JSON string
     */
    std::string generate_dicom_details(const std::string& dicom_id,
                                       const std::string& patient_id,
                                       const std::string& modality,
                                       const std::string& user_id);
    
    /**
     * Generate deletion details JSON
     * 
     * @param dicom_id DICOM identifier
     * @param permanent Whether deletion was permanent
     * @return JSON string
     */
    std::string generate_deletion_details(const std::string& dicom_id, bool permanent);
    
    /**
     * Generate waveform extraction details JSON
     * 
     * @param dicom_id DICOM identifier
     * @param patient_id Patient identifier
     * @param num_channels Number of channels
     * @param num_samples Number of samples
     * @return JSON string
     */
    std::string generate_waveform_details(const std::string& dicom_id,
                                          const std::string& patient_id,
                                          int num_channels,
                                          int num_samples);
    
    /**
     * Generate access denied details JSON
     * 
     * @param user_id User identifier
     * @param resource Resource being accessed
     * @param reason Reason for denial
     * @return JSON string
     */
    std::string generate_access_details(const std::string& user_id,
                                        const std::string& resource,
                                        const std::string& reason);
    
    /**
     * Generate error details JSON
     * 
     * @param operation Operation that failed
     * @param error_message Error description
     * @param context Additional context
     * @return JSON string
     */
    std::string generate_error_details(const std::string& operation,
                                      const std::string& error_message,
                                      const std::string& context);
    
    /**
     * Generate statistics JSON
     * 
     * @return JSON string with current statistics
     */
    std::string generate_statistics_json();
    
    /**
     * Write audit event to log file
     * 
     * @param event Audit event to write
     */
    void write_audit_event(const AuditEvent& event);
    
    /**
     * Update internal statistics
     * 
     * @param event_type Type of event
     */
    void update_statistics(const std::string& event_type);
    
    /**
     * Parse audit line from log file
     * 
     * @param line Log line to parse
     * @return Parsed audit event
     */
    AuditEvent parse_audit_line(const std::string& line);
};

} // namespace dicom
} // namespace vitalstream
