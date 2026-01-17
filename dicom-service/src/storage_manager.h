#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <mutex>
#include <chrono>

namespace vitalstream {
namespace dicom {

/**
 * Storage statistics structure
 */
struct StorageStats {
    size_t total_files;
    size_t total_size;
    std::chrono::system_clock::time_point last_cleanup;
    
    StorageStats() : total_files(0), total_size(0) {
        last_cleanup = std::chrono::system_clock::now();
    }
};

/**
 * Storage Manager for DICOM files
 * 
 * Handles file storage, retrieval, and management with proper
 * directory structure and performance optimization.
 */
class StorageManager {
public:
    explicit StorageManager(const std::string& base_path);
    ~StorageManager();
    
    /**
     * Store DICOM file with proper directory structure
     * 
     * @param dicom_id Unique DICOM identifier
     * @param study_uid Study instance UID
     * @param series_uid Series instance UID
     * @param data File data
     * @param file_path Output parameter for stored file path
     * @return true if successful
     */
    bool store_file(const std::string& dicom_id, 
                   const std::string& study_uid,
                   const std::string& series_uid,
                   const std::vector<uint8_t>& data,
                   std::string& file_path);
    
    /**
     * Retrieve DICOM file data
     * 
     * @param dicom_id Unique DICOM identifier
     * @param data Output parameter for file data
     * @return true if successful
     */
    bool retrieve_file(const std::string& dicom_id, std::vector<uint8_t>& data);
    
    /**
     * Delete DICOM file
     * 
     * @param dicom_id Unique DICOM identifier
     * @param permanent If false, move to trash; if true, delete permanently
     * @return true if successful
     */
    bool delete_file(const std::string& dicom_id, bool permanent = false);
    
    /**
     * List DICOM files
     * 
     * @param study_uid Optional study UID filter
     * @return List of DICOM IDs
     */
    std::vector<std::string> list_files(const std::string& study_uid = "");
    
    /**
     * Get storage statistics
     * 
     * @return Storage statistics
     */
    StorageStats get_statistics();
    
    /**
     * Clean up temporary files older than 24 hours
     * 
     * @return true if successful
     */
    bool cleanup_temp_files();
    
    /**
     * Verify storage integrity
     * 
     * @return true if all files are valid
     */
    bool verify_storage_integrity();
    
    /**
     * Get storage efficiency (0.0 to 1.0)
     * 
     * @return Efficiency score
     */
    double get_storage_efficiency();

private:
    std::string base_path_;
    StorageStats stats_;
    mutable std::mutex stats_mutex_;
    
    /**
     * Scan existing storage and update statistics
     */
    void scan_storage();
    
    /**
     * Find file path for DICOM ID
     * 
     * @param dicom_id Unique DICOM identifier
     * @return File path or empty string if not found
     */
    std::string find_file_path(const std::string& dicom_id);
};

} // namespace dicom
} // namespace vitalstream
