#include "storage_manager.h"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace vitalstream {
namespace dicom {

StorageManager::StorageManager(const std::string& base_path) 
    : base_path_(base_path) {
    spdlog::info("📁 Initializing Storage Manager with base path: {}", base_path);
    
    // Create base directory if it doesn't exist
    std::filesystem::create_directories(base_path_);
    
    // Create subdirectories
    std::filesystem::create_directories(base_path_ + "/studies");
    std::filesystem::create_directories(base_path_ + "/temp");
    std::filesystem::create_directories(base_path_ + "/trash");
    
    // Initialize statistics
    stats_.total_files = 0;
    stats_.total_size = 0;
    stats_.last_cleanup = std::chrono::system_clock::now();
    
    // Scan existing files
    scan_storage();
}

StorageManager::~StorageManager() {
    spdlog::info("📁 Storage Manager shutting down");
}

bool StorageManager::store_file(const std::string& dicom_id, 
                               const std::string& study_uid,
                               const std::string& series_uid,
                               const std::vector<uint8_t>& data,
                               std::string& file_path) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Create study directory structure
        std::string study_path = base_path_ + "/studies/" + study_uid;
        std::string series_path = study_path + "/" + series_uid;
        std::filesystem::create_directories(series_path);
        
        // Generate file path
        file_path = series_path + "/" + dicom_id + ".dcm";
        
        // Check if file already exists
        if (std::filesystem::exists(file_path)) {
            spdlog::warn("⚠️ DICOM file already exists: {}", file_path);
            return false;
        }
        
        // Write file
        std::ofstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            spdlog::error("❌ Failed to create file: {}", file_path);
            return false;
        }
        
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        file.close();
        
        // Verify file was written correctly
        if (!std::filesystem::exists(file_path)) {
            spdlog::error("❌ File verification failed: {}", file_path);
            return false;
        }
        
        // Update statistics
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_files++;
            stats_.total_size += data.size();
        }
        
        // Calculate storage time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        ).count();
        
        spdlog::info("✅ DICOM file stored: {} ({}ms, {}KB)", 
                    dicom_id, duration, data.size() / 1024);
        
        return true;
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Exception storing file {}: {}", dicom_id, e.what());
        return false;
    }
}

bool StorageManager::retrieve_file(const std::string& dicom_id, 
                                  std::vector<uint8_t>& data) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Find file in storage
        std::string file_path = find_file_path(dicom_id);
        if (file_path.empty()) {
            spdlog::error("❌ DICOM file not found: {}", dicom_id);
            return false;
        }
        
        // Read file
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            spdlog::error("❌ Failed to open file: {}", file_path);
            return false;
        }
        
        // Get file size
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        // Read data
        data.resize(file_size);
        file.read(reinterpret_cast<char*>(data.data()), file_size);
        file.close();
        
        // Calculate retrieval time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        ).count();
        
        spdlog::info("✅ DICOM file retrieved: {} ({}ms, {}KB)", 
                    dicom_id, duration, data.size() / 1024);
        
        return true;
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Exception retrieving file {}: {}", dicom_id, e.what());
        return false;
    }
}

bool StorageManager::delete_file(const std::string& dicom_id, bool permanent) {
    try {
        std::string file_path = find_file_path(dicom_id);
        if (file_path.empty()) {
            spdlog::error("❌ DICOM file not found for deletion: {}", dicom_id);
            return false;
        }
        
        if (permanent) {
            // Get file size before deletion
            size_t file_size = std::filesystem::file_size(file_path);
            
            // Delete permanently
            if (std::filesystem::remove(file_path)) {
                // Update statistics
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.total_files = std::max(0, (int)stats_.total_files - 1);
                    stats_.total_size = std::max(0ULL, (unsigned long long)(stats_.total_size - file_size));
                }
                
                spdlog::info("🗑️ DICOM file permanently deleted: {}", dicom_id);
                return true;
            }
        } else {
            // Move to trash
            std::string trash_path = base_path_ + "/trash/" + dicom_id + ".dcm";
            std::error_code ec;
            std::filesystem::rename(file_path, trash_path, ec);
            if (!ec) {
                spdlog::info("🗑️ DICOM file moved to trash: {}", dicom_id);
                return true;
            }
        }
        
        return false;
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Exception deleting file {}: {}", dicom_id, e.what());
        return false;
    }
}

std::vector<std::string> StorageManager::list_files(const std::string& study_uid) {
    std::vector<std::string> files;
    
    try {
        std::string study_path = base_path_ + "/studies";
        if (!study_uid.empty()) {
            study_path += "/" + study_uid;
        }
        
        if (!std::filesystem::exists(study_path)) {
            return files;
        }
        
        for (const auto& entry : std::filesystem::recursive_directory_iterator(study_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".dcm") {
                files.push_back(entry.path().stem());
            }
        }
        
        spdlog::info("📋 Listed {} DICOM files", files.size());
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Exception listing files: {}", e.what());
    }
    
    return files;
}

StorageStats StorageManager::get_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

bool StorageManager::cleanup_temp_files() {
    try {
        std::string temp_path = base_path_ + "/temp";
        if (!std::filesystem::exists(temp_path)) {
            return true;
        }
        
        size_t cleaned_files = 0;
        size_t cleaned_size = 0;
        
        auto cutoff_time = std::chrono::system_clock::now() - std::chrono::hours(24);
        
        for (const auto& entry : std::filesystem::directory_iterator(temp_path)) {
            if (entry.is_regular_file()) {
                auto file_time = std::filesystem::last_write_time(entry.path());
                
                // Convert filesystem time to system_clock time for comparison
                auto file_time_sys = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                    file_time - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
                
                if (file_time_sys < cutoff_time) {
                    size_t file_size = std::filesystem::file_size(entry.path());
                    if (std::filesystem::remove(entry.path())) {
                        cleaned_files++;
                        cleaned_size += file_size;
                    }
                }
            }
        }
        
        spdlog::info("🧹 Cleaned up {} temp files ({}KB)", 
                    cleaned_files, cleaned_size / 1024);
        
        // Update last cleanup time
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.last_cleanup = std::chrono::system_clock::now();
        }
        
        return true;
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Exception during cleanup: {}", e.what());
        return false;
    }
}

bool StorageManager::verify_storage_integrity() {
    try {
        size_t corrupted_files = 0;
        size_t total_files = 0;
        
        std::string studies_path = base_path_ + "/studies";
        if (!std::filesystem::exists(studies_path)) {
            return true;
        }
        
        for (const auto& entry : std::filesystem::recursive_directory_iterator(studies_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".dcm") {
                total_files++;
                
                // Try to read DICOM header
                std::ifstream file(entry.path(), std::ios::binary);
                if (file.is_open()) {
                    // Read first few bytes to check DICOM signature
                    char header[128];
                    file.read(header, 128);
                    file.close();
                    
                    // Check for DICOM signature (should start with "DICM" at offset 128)
                    if (file.tellg() < 132 || std::string(header + 128, 4) != "DICM") {
                        spdlog::warn("⚠️ Corrupted DICOM file detected: {}", entry.path().string());
                        corrupted_files++;
                    }
                } else {
                    spdlog::warn("⚠️ Cannot read DICOM file: {}", entry.path().string());
                    corrupted_files++;
                }
            }
        }
        
        spdlog::info("🔍 Storage integrity check: {} files, {} corrupted", 
                    total_files, corrupted_files);
        
        return corrupted_files == 0;
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Exception during integrity check: {}", e.what());
        return false;
    }
}

double StorageManager::get_storage_efficiency() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (stats_.total_files == 0) {
        return 0.0;
    }
    
    // Calculate average file size
    double avg_size = static_cast<double>(stats_.total_size) / stats_.total_files;
    
    // Efficiency based on average file size (optimal around 1MB)
    if (avg_size < 1024 * 1024) { // Less than 1MB
        return avg_size / (1024 * 1024); // 0.0 to 1.0
    } else if (avg_size < 10 * 1024 * 1024) { // 1MB to 10MB
        return 1.0; // Optimal
    } else { // Larger than 10MB
        return 1.0 / (avg_size / (10 * 1024 * 1024)); // Decreasing efficiency
    }
}

// Private methods

void StorageManager::scan_storage() {
    try {
        size_t total_files = 0;
        size_t total_size = 0;
        
        std::string studies_path = base_path_ + "/studies";
        if (!std::filesystem::exists(studies_path)) {
            return;
        }
        
        for (const auto& entry : std::filesystem::recursive_directory_iterator(studies_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".dcm") {
                total_files++;
                total_size += std::filesystem::file_size(entry.path());
            }
        }
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_files = total_files;
            stats_.total_size = total_size;
        }
        
        spdlog::info("📊 Storage scan completed: {} files, {}MB", 
                    total_files, total_size / (1024 * 1024));
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Exception during storage scan: {}", e.what());
    }
}

std::string StorageManager::find_file_path(const std::string& dicom_id) {
    try {
        std::string studies_path = base_path_ + "/studies";
        if (!std::filesystem::exists(studies_path)) {
            return "";
        }
        
        for (const auto& entry : std::filesystem::recursive_directory_iterator(studies_path)) {
            if (entry.is_regular_file() && 
                entry.path().extension() == ".dcm" &&
                entry.path().stem() == dicom_id) {
                return entry.path();
            }
        }
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Exception finding file path: {}", e.what());
    }
    
    return "";
}

} // namespace dicom
} // namespace vitalstream
