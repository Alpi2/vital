#include "audit_logger.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <mutex>

namespace vitalstream {
namespace dicom {

AuditLogger::AuditLogger(const std::string& log_file_path) 
    : log_file_path_(log_file_path) {
    spdlog::info("📋 Initializing Audit Logger: {}", log_file_path);
    
    // Create log directory if it doesn't exist
    size_t last_slash = log_file_path_.find_last_of('/');
    if (last_slash != std::string::npos) {
        std::string log_dir = log_file_path_.substr(0, last_slash);
        std::filesystem::create_directories(log_dir);
    }
    
    // Initialize statistics
    stats_.total_operations = 0;
    stats_.dicom_stored = 0;
    stats_.dicom_retrieved = 0;
    stats_.dicom_deleted = 0;
    stats_.waveform_extracted = 0;
    stats_.errors = 0;
    stats_.start_time = std::chrono::system_clock::now();
}

AuditLogger::~AuditLogger() {
    spdlog::info("📋 Audit Logger shutting down");
    
    // Log final statistics
    AuditEvent summary_event;
    summary_event.timestamp = get_timestamp();
    summary_event.event_type = "AUDIT_SUMMARY";
    summary_event.user_id = "system";
    summary_event.patient_id = "";
    summary_event.dicom_id = "";
    summary_event.description = "Audit logger shutdown";
    summary_event.details = generate_statistics_json();
    write_audit_event(summary_event);
}

void AuditLogger::log_dicom_stored(const std::string& dicom_id, 
                                   const std::string& patient_id, 
                                   const std::string& modality) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    auto timestamp = get_timestamp();
    
    AuditEvent event;
    event.timestamp = timestamp;
    event.event_type = "DICOM_STORED";
    event.user_id = "system"; // TODO: Get from context
    event.patient_id = patient_id;
    event.dicom_id = dicom_id;
    event.description = "DICOM file stored successfully";
    event.details = generate_dicom_details(dicom_id, patient_id, modality, "");
    
    write_audit_event(event);
    update_statistics("DICOM_STORED");
    
    spdlog::info("📋 Audit: DICOM_STORED - ID: {}, Patient: {}, Modality: {}", 
                dicom_id, patient_id, modality);
}

void AuditLogger::log_dicom_retrieved(const std::string& dicom_id, 
                                     const std::string& patient_id) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    auto timestamp = get_timestamp();
    
    AuditEvent event;
    event.timestamp = timestamp;
    event.event_type = "DICOM_RETRIEVED";
    event.user_id = "system"; // TODO: Get from context
    event.patient_id = patient_id;
    event.dicom_id = dicom_id;
    event.description = "DICOM file retrieved";
    event.details = generate_dicom_details(dicom_id, patient_id, "", "");
    
    write_audit_event(event);
    update_statistics("DICOM_RETRIEVED");
    
    spdlog::info("📋 Audit: DICOM_RETRIEVED - ID: {}, Patient: {}", dicom_id, patient_id);
}

void AuditLogger::log_dicom_deleted(const std::string& dicom_id, bool permanent) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    auto timestamp = get_timestamp();
    
    AuditEvent event;
    event.timestamp = timestamp;
    event.event_type = "DICOM_DELETED";
    event.user_id = "system"; // TODO: Get from context
    event.dicom_id = dicom_id;
    event.description = permanent ? "DICOM file permanently deleted" : "DICOM file moved to trash";
    event.details = generate_deletion_details(dicom_id, permanent);
    
    write_audit_event(event);
    update_statistics("DICOM_DELETED");
    
    spdlog::info("📋 Audit: DICOM_DELETED - ID: {}, Permanent: {}", dicom_id, permanent);
}

void AuditLogger::log_waveform_extracted(const std::string& dicom_id, 
                                        const std::string& patient_id,
                                        int num_channels,
                                        int num_samples) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    auto timestamp = get_timestamp();
    
    AuditEvent event;
    event.timestamp = timestamp;
    event.event_type = "WAVEFORM_EXTRACTED";
    event.user_id = "system"; // TODO: Get from context
    event.patient_id = patient_id;
    event.dicom_id = dicom_id;
    event.description = "ECG waveform extracted from DICOM";
    event.details = generate_waveform_details(dicom_id, patient_id, num_channels, num_samples);
    
    write_audit_event(event);
    update_statistics("WAVEFORM_EXTRACTED");
    
    spdlog::info("📋 Audit: WAVEFORM_EXTRACTED - ID: {}, Channels: {}, Samples: {}", 
                dicom_id, num_channels, num_samples);
}

void AuditLogger::log_access_denied(const std::string& user_id, 
                                   const std::string& resource,
                                   const std::string& reason) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    auto timestamp = get_timestamp();
    
    AuditEvent event;
    event.timestamp = timestamp;
    event.event_type = "ACCESS_DENIED";
    event.user_id = user_id;
    event.description = "Access denied to resource";
    event.details = generate_access_details(user_id, resource, reason);
    
    write_audit_event(event);
    update_statistics("ACCESS_DENIED");
    
    spdlog::warn("📋 Audit: ACCESS_DENIED - User: {}, Resource: {}, Reason: {}", 
                 user_id, resource, reason);
}

void AuditLogger::log_error(const std::string& operation, 
                           const std::string& error_message,
                           const std::string& context) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    auto timestamp = get_timestamp();
    
    AuditEvent event;
    event.timestamp = timestamp;
    event.event_type = "ERROR";
    event.user_id = "system";
    event.description = "Error occurred during operation";
    event.details = generate_error_details(operation, error_message, context);
    
    write_audit_event(event);
    update_statistics("ERROR");
    
    spdlog::error("📋 Audit: ERROR - Operation: {}, Error: {}", operation, error_message);
}

void AuditLogger::log_system_event(const std::string& event_type, 
                                  const std::string& description,
                                  const std::string& details) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    auto timestamp = get_timestamp();
    
    AuditEvent event;
    event.timestamp = timestamp;
    event.event_type = event_type;
    event.user_id = "system";
    event.description = description;
    event.details = details;
    
    write_audit_event(event);
    
    spdlog::info("📋 Audit: {} - {}", event_type, description);
}

AuditStats AuditLogger::get_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

std::vector<AuditEvent> AuditLogger::query_events(const std::string& event_type,
                                                  const std::string& user_id,
                                                  const std::chrono::system_clock::time_point& from_time,
                                                  const std::chrono::system_clock::time_point& to_time,
                                                  int limit) {
    std::vector<AuditEvent> events;
    
    try {
        std::ifstream log_file(log_file_path_);
        if (!log_file.is_open()) {
            spdlog::error("❌ Cannot open audit log file for querying: {}", log_file_path_);
            return events;
        }
        
        std::string line;
        while (std::getline(log_file, line) && events.size() < limit) {
            try {
                AuditEvent event = parse_audit_line(line);
                
                // Apply filters
                if (!event_type.empty() && event.event_type != event_type) {
                    continue;
                }
                
                if (!user_id.empty() && event.user_id != user_id) {
                    continue;
                }
                
                // TODO: Add time filtering
                
                events.push_back(event);
                
            } catch (const std::exception& e) {
                spdlog::warn("⚠️ Failed to parse audit line: {}", e.what());
            }
        }
        
        log_file.close();
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Exception querying audit events: {}", e.what());
    }
    
    return events;
}

// Private methods

std::string AuditLogger::get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
    
    return ss.str();
}

std::string AuditLogger::generate_dicom_details(const std::string& dicom_id,
                                               const std::string& patient_id,
                                               const std::string& modality,
                                               const std::string& user_id) {
    std::stringstream ss;
    ss << "{"
       << "\"dicom_id\":\"" << dicom_id << "\","
       << "\"patient_id\":\"" << patient_id << "\","
       << "\"modality\":\"" << modality << "\","
       << "\"user_id\":\"" << user_id << "\""
       << "}";
    return ss.str();
}

std::string AuditLogger::generate_deletion_details(const std::string& dicom_id, bool permanent) {
    std::stringstream ss;
    ss << "{"
       << "\"dicom_id\":\"" << dicom_id << "\","
       << "\"permanent\":" << (permanent ? "true" : "false")
       << "}";
    return ss.str();
}

std::string AuditLogger::generate_waveform_details(const std::string& dicom_id,
                                                  const std::string& patient_id,
                                                  int num_channels,
                                                  int num_samples) {
    std::stringstream ss;
    ss << "{"
       << "\"dicom_id\":\"" << dicom_id << "\","
       << "\"patient_id\":\"" << patient_id << "\","
       << "\"num_channels\":" << num_channels << ","
       << "\"num_samples\":" << num_samples
       << "}";
    return ss.str();
}

std::string AuditLogger::generate_access_details(const std::string& user_id,
                                                const std::string& resource,
                                                const std::string& reason) {
    std::stringstream ss;
    ss << "{"
       << "\"user_id\":\"" << user_id << "\","
       << "\"resource\":\"" << resource << "\","
       << "\"reason\":\"" << reason << "\""
       << "}";
    return ss.str();
}

std::string AuditLogger::generate_error_details(const std::string& operation,
                                              const std::string& error_message,
                                              const std::string& context) {
    std::stringstream ss;
    ss << "{"
       << "\"operation\":\"" << operation << "\","
       << "\"error_message\":\"" << error_message << "\","
       << "\"context\":\"" << context << "\""
       << "}";
    return ss.str();
}

std::string AuditLogger::generate_statistics_json() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    std::stringstream ss;
    ss << "{"
       << "\"total_operations\":" << stats_.total_operations << ","
       << "\"dicom_stored\":" << stats_.dicom_stored << ","
       << "\"dicom_retrieved\":" << stats_.dicom_retrieved << ","
       << "\"dicom_deleted\":" << stats_.dicom_deleted << ","
       << "\"waveform_extracted\":" << stats_.waveform_extracted << ","
       << "\"errors\":" << stats_.errors
       << "}";
    return ss.str();
}

void AuditLogger::write_audit_event(const AuditEvent& event) {
    try {
        std::ofstream log_file(log_file_path_, std::ios::app);
        if (!log_file.is_open()) {
            spdlog::error("❌ Cannot open audit log file: {}", log_file_path_);
            return;
        }
        
        // Write audit event in JSON format
        log_file << event.timestamp << " "
                 << event.event_type << " "
                 << event.user_id << " "
                 << event.patient_id << " "
                 << event.dicom_id << " "
                 << event.description << " "
                 << event.details << std::endl;
        
        log_file.close();
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Exception writing audit event: {}", e.what());
    }
}

void AuditLogger::update_statistics(const std::string& event_type) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats_.total_operations++;
    
    if (event_type == "DICOM_STORED") {
        stats_.dicom_stored++;
    } else if (event_type == "DICOM_RETRIEVED") {
        stats_.dicom_retrieved++;
    } else if (event_type == "DICOM_DELETED") {
        stats_.dicom_deleted++;
    } else if (event_type == "WAVEFORM_EXTRACTED") {
        stats_.waveform_extracted++;
    } else if (event_type == "ACCESS_DENIED") {
        stats_.access_denied++;
    } else if (event_type == "ERROR") {
        stats_.errors++;
    }
}

AuditEvent AuditLogger::parse_audit_line(const std::string& line) {
    AuditEvent event;
    
    // Simple parsing - split by spaces
    std::istringstream iss(line);
    std::string token;
    
    // Timestamp
    if (std::getline(iss, token, ' ')) {
        event.timestamp = token;
    }
    
    // Event type
    if (std::getline(iss, token, ' ')) {
        event.event_type = token;
    }
    
    // User ID
    if (std::getline(iss, token, ' ')) {
        event.user_id = token;
    }
    
    // Patient ID
    if (std::getline(iss, token, ' ')) {
        event.patient_id = token;
    }
    
    // DICOM ID
    if (std::getline(iss, token, ' ')) {
        event.dicom_id = token;
    }
    
    // Description (might contain spaces)
    std::string description_part;
    while (std::getline(iss, token, ' ')) {
        if (token.find('{') == 0) {
            // This is the JSON details part
            event.details = token;
            break;
        }
        description_part += token + " ";
    }
    event.description = description_part;
    
    return event;
}

} // namespace dicom
} // namespace vitalstream
