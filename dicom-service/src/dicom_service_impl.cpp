#include "dicom_service_impl.h"
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcmetinf.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <uuid/uuid.h>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace vitalstream {
namespace dicom {

DICOMServiceImpl::DICOMServiceImpl(const std::string& storage_path, 
                                   std::shared_ptr<AuditLogger> audit_logger)
    : storage_path_(storage_path), audit_logger_(audit_logger) {
    spdlog::info("🏥 Initializing DICOM Service with storage path: {}", storage_path);
    
    // Create storage directory if it doesn't exist
    std::filesystem::create_directories(storage_path_);
    
    // Initialize logger if not provided
    if (!audit_logger_) {
        audit_logger_ = std::make_shared<AuditLogger>();
    }
}

grpc::Status DICOMServiceImpl::StoreDICOM(
    grpc::ServerContext* context,
    const ::dicom::v1::StoreDICOMRequest* request,
    ::dicom::v1::StoreDICOMResponse* response
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    spdlog::info("🏥 Storing DICOM file: {} bytes", request->file_data().size());
    
    try {
        // Validate input
        if (request->file_data().empty()) {
            spdlog::error("❌ Empty DICOM file data");
            return grpc::Status(
                grpc::StatusCode::INVALID_ARGUMENT,
                "DICOM file data cannot be empty"
            );
        }
        
        if (request->file_data().size() > 100 * 1024 * 1024) { // 100MB limit
            spdlog::error("❌ DICOM file too large: {} bytes", request->file_data().size());
            return grpc::Status(
                grpc::StatusCode::INVALID_ARGUMENT,
                "DICOM file too large (max 100MB)"
            );
        }
        
        // Parse DICOM data using DCMTK - write to temp file first
        std::string temp_file = "/tmp/temp_dicom_" + std::to_string(getpid()) + ".dcm";
        std::ofstream temp_stream(temp_file, std::ios::binary);
        temp_stream.write(request->file_data().data(), request->file_data().size());
        temp_stream.close();
        
        DcmFileFormat fileformat;
        OFCondition status = fileformat.loadFile(temp_file.c_str());
        
        // Clean up temp file
        std::filesystem::remove(temp_file);
        
        if (status.bad()) {
            spdlog::error("❌ Failed to parse DICOM: {}", status.text());
            return grpc::Status(
                grpc::StatusCode::INVALID_ARGUMENT,
                std::string("Invalid DICOM file: ") + status.text()
            );
        }
        
        // Validate DICOM compliance
        if (!validate_dicom_compliance(fileformat)) {
            return grpc::Status(
                grpc::StatusCode::INVALID_ARGUMENT,
                "DICOM file does not comply with standard"
            );
        }
        
        // Extract metadata
        DcmDataset* dataset = fileformat.getDataset();
        auto metadata = extract_metadata(dataset);
        
        // Generate unique DICOM ID
        std::string dicom_id = generate_uuid();
        
        // Create storage directory structure
        std::string study_dir = storage_path_ + "/" + metadata.study_instance_uid();
        std::filesystem::create_directories(study_dir);
        
        // Save to storage
        std::string file_path = study_dir + "/" + dicom_id + ".dcm";
        status = fileformat.saveFile(file_path.c_str());
        
        if (status.bad()) {
            spdlog::error("❌ Failed to save DICOM: {}", status.text());
            return grpc::Status(
                grpc::StatusCode::INTERNAL,
                std::string("Failed to save DICOM file: ") + status.text()
            );
        }
        
        // Build response
        response->set_dicom_id(dicom_id);
        response->set_patient_id(metadata.patient_id());
        response->set_patient_name(metadata.patient_name());
        response->set_study_date(metadata.study_date());
        response->set_modality(metadata.modality());
        response->set_file_path(file_path);
        response->set_file_size(request->file_data().size());
        response->set_study_instance_uid(metadata.study_instance_uid());
        response->set_series_instance_uid(metadata.series_instance_uid());
        response->set_sop_instance_uid(metadata.sop_instance_uid());
        
        // Set creation timestamp
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()
        ).count();
        
        google::protobuf::Timestamp* created_at = new google::protobuf::Timestamp();
        created_at->set_seconds(timestamp / 1000);
        created_at->set_nanos((timestamp % 1000) * 1000000);
        response->set_allocated_created_at(created_at);
        
        // Extract waveform if ECG
        if (metadata.modality() == "ECG") {
            auto waveform = extract_ecg_waveform(dataset);
            if (waveform.has_value()) {
                *response->mutable_waveform_data() = waveform.value();
                spdlog::info("📈 ECG waveform extracted: {} channels, {} samples", 
                           waveform->num_channels(), waveform->num_samples());
            }
        }
        
        // Audit logging
        audit_logger_->log_dicom_stored(dicom_id, metadata.patient_id(), metadata.modality());
        
        // Calculate processing time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        ).count();
        
        spdlog::info("✅ DICOM stored successfully: {} ({}ms)", dicom_id, duration);
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Exception storing DICOM: {}", e.what());
        return grpc::Status(
            grpc::StatusCode::INTERNAL,
            std::string("Internal error: ") + e.what()
        );
    }
}

grpc::Status DICOMServiceImpl::GetDICOM(
    grpc::ServerContext* context,
    const ::dicom::v1::GetDICOMRequest* request,
    ::dicom::v1::GetDICOMResponse* response
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    spdlog::info("📥 Retrieving DICOM: {}", request->dicom_id());
    
    try {
        if (request->dicom_id().empty()) {
            return grpc::Status(
                grpc::StatusCode::INVALID_ARGUMENT,
                "DICOM ID cannot be empty"
            );
        }
        
        // Find DICOM file in storage
        std::string file_path;
        bool found = false;
        
        // Search through study directories
        for (const auto& entry : std::filesystem::directory_iterator(storage_path_)) {
            if (entry.is_directory()) {
                std::string potential_path = entry.path() / (request->dicom_id() + ".dcm");
                if (std::filesystem::exists(potential_path)) {
                    file_path = potential_path;
                    found = true;
                    break;
                }
            }
        }
        
        if (!found) {
            spdlog::error("❌ DICOM file not found: {}", request->dicom_id());
            return grpc::Status(
                grpc::StatusCode::NOT_FOUND,
                "DICOM file not found"
            );
        }
        
        // Read file
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            spdlog::error("❌ Failed to open DICOM file: {}", file_path);
            return grpc::Status(
                grpc::StatusCode::INTERNAL,
                "Failed to open DICOM file"
            );
        }
        
        // Read file data
        std::string file_data(
            (std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>()
        );
        
        response->set_file_data(file_data);
        response->set_dicom_id(request->dicom_id());
        
        // Include metadata if requested
        if (request->include_metadata()) {
            DcmFileFormat fileformat;
            OFCondition status = fileformat.loadFile(file_path.c_str());
            
            if (status.good()) {
                DcmDataset* dataset = fileformat.getDataset();
                auto metadata = extract_metadata(dataset);
                
                auto* metadata_response = response->mutable_metadata();
                metadata_response->set_patient_id(metadata.patient_id());
                metadata_response->set_patient_name(metadata.patient_name());
                metadata_response->set_study_date(metadata.study_date());
                metadata_response->set_modality(metadata.modality());
                metadata_response->set_study_instance_uid(metadata.study_instance_uid());
                metadata_response->set_series_instance_uid(metadata.series_instance_uid());
                metadata_response->set_sop_instance_uid(metadata.sop_instance_uid());
                
                auto now = std::chrono::system_clock::now();
                auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()
                ).count();
                
                google::protobuf::Timestamp* created_at = new google::protobuf::Timestamp();
                created_at->set_seconds(timestamp / 1000);
                created_at->set_nanos((timestamp % 1000) * 1000000);
                metadata_response->set_allocated_created_at(created_at);
            }
        }
        
        // Calculate processing time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        ).count();
        
        spdlog::info("✅ DICOM retrieved: {} ({}ms, {}KB)", 
                    request->dicom_id(), duration, file_data.size() / 1024);
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Exception retrieving DICOM: {}", e.what());
        return grpc::Status(
            grpc::StatusCode::INTERNAL,
            std::string("Internal error: ") + e.what()
        );
    }
}

grpc::Status DICOMServiceImpl::ExtractWaveform(
    grpc::ServerContext* context,
    const ::dicom::v1::ExtractWaveformRequest* request,
    ::dicom::v1::ExtractWaveformResponse* response
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    spdlog::info("📈 Extracting waveform from DICOM: {}", request->dicom_id());
    
    try {
        if (request->dicom_id().empty()) {
            return grpc::Status(
                grpc::StatusCode::INVALID_ARGUMENT,
                "DICOM ID cannot be empty"
            );
        }
        
        // Find DICOM file
        std::string file_path;
        bool found = false;
        
        for (const auto& entry : std::filesystem::directory_iterator(storage_path_)) {
            if (entry.is_directory()) {
                std::string potential_path = entry.path() / (request->dicom_id() + ".dcm");
                if (std::filesystem::exists(potential_path)) {
                    file_path = potential_path;
                    found = true;
                    break;
                }
            }
        }
        
        if (!found) {
            return grpc::Status(
                grpc::StatusCode::NOT_FOUND,
                "DICOM file not found"
            );
        }
        
        // Load DICOM file
        DcmFileFormat fileformat;
        OFCondition status = fileformat.loadFile(file_path.c_str());
        
        if (status.bad()) {
            return grpc::Status(
                grpc::StatusCode::INTERNAL,
                "Failed to load DICOM file"
            );
        }
        
        // Extract waveform
        DcmDataset* dataset = fileformat.getDataset();
        auto waveform = extract_ecg_waveform(dataset);
        
        if (!waveform.has_value()) {
            return grpc::Status(
                grpc::StatusCode::NOT_FOUND,
                "No waveform data found in DICOM file"
            );
        }
        
        // Filter channels if specified
        if (!request->channel_names().empty()) {
            // TODO: Implement channel filtering
            spdlog::warn("⚠️ Channel filtering not yet implemented");
        }
        
        *response->mutable_waveform_data() = waveform.value();
        response->set_dicom_id(request->dicom_id());
        
        // Calculate processing time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        ).count();
        
        spdlog::info("✅ Waveform extracted: {} ({}ms)", request->dicom_id(), duration);
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Exception extracting waveform: {}", e.what());
        return grpc::Status(
            grpc::StatusCode::INTERNAL,
            std::string("Internal error: ") + e.what()
        );
    }
}

grpc::Status DICOMServiceImpl::QueryDICOM(
    grpc::ServerContext* context,
    const ::dicom::v1::QueryDICOMRequest* request,
    ::dicom::v1::QueryDICOMResponse* response
) {
    spdlog::info("🔍 Querying DICOM files");
    
    try {
        auto results = query_files(
            request->patient_id(),
            request->modality(),
            request->date_from(),
            request->date_to(),
            request->page_size(),
            request->page_token()
        );
        
        for (const auto& result : results) {
            auto* dicom_file = response->add_dicom_files();
            dicom_file->CopyFrom(result);
        }
        
        response->set_total_count(results.size());
        
        spdlog::info("✅ DICOM query completed: {} results", results.size());
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Exception querying DICOM: {}", e.what());
        return grpc::Status(
            grpc::StatusCode::INTERNAL,
            std::string("Internal error: ") + e.what()
        );
    }
}

grpc::Status DICOMServiceImpl::DeleteDICOM(
    grpc::ServerContext* context,
    const ::dicom::v1::DeleteDICOMRequest* request,
    ::dicom::v1::DeleteDICOMResponse* response
) {
    spdlog::info("🗑️ Deleting DICOM: {} (permanent={})", 
                request->dicom_id(), request->permanent());
    
    try {
        if (request->dicom_id().empty()) {
            return grpc::Status(
                grpc::StatusCode::INVALID_ARGUMENT,
                "DICOM ID cannot be empty"
            );
        }
        
        // Find and delete file
        bool found = false;
        for (const auto& entry : std::filesystem::directory_iterator(storage_path_)) {
            if (entry.is_directory()) {
                std::string file_path = entry.path() / (request->dicom_id() + ".dcm");
                if (std::filesystem::exists(file_path)) {
                    if (request->permanent()) {
                        std::filesystem::remove(file_path);
                    } else {
                        // TODO: Move to trash directory
                        std::filesystem::remove(file_path);
                    }
                    found = true;
                    break;
                }
            }
        }
        
        if (!found) {
            response->set_success(false);
            response->set_message("DICOM file not found");
        } else {
            response->set_success(true);
            response->set_message("DICOM file deleted successfully");
            
            // Audit logging
            audit_logger_->log_dicom_deleted(request->dicom_id(), request->permanent());
        }
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Exception deleting DICOM: {}", e.what());
        return grpc::Status(
            grpc::StatusCode::INTERNAL,
            std::string("Internal error: ") + e.what()
        );
    }
}

grpc::Status DICOMServiceImpl::ListDICOM(
    grpc::ServerContext* context,
    const ::dicom::v1::ListDICOMRequest* request,
    ::dicom::v1::ListDICOMResponse* response
) {
    spdlog::info("📋 Listing DICOM files");
    
    try {
        // TODO: Implement pagination
        auto results = query_files(
            request->patient_id(),
            request->modality(),
            google::protobuf::Timestamp(), // No date filter
            google::protobuf::Timestamp(),
            request->page_size(),
            request->page_token()
        );
        
        for (const auto& result : results) {
            auto* dicom_file = response->add_dicom_files();
            dicom_file->CopyFrom(result);
        }
        
        response->set_total_count(results.size());
        
        spdlog::info("✅ DICOM list completed: {} results", results.size());
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Exception listing DICOM: {}", e.what());
        return grpc::Status(
            grpc::StatusCode::INTERNAL,
            std::string("Internal error: ") + e.what()
        );
    }
}

grpc::Status DICOMServiceImpl::HealthCheck(
    grpc::ServerContext* context,
    const common::v1::HealthCheckRequest* request,
    common::v1::HealthCheckResponse* response
) {
    spdlog::debug("🏥 Health check requested");
    
    try {
        // Check storage accessibility
        if (!std::filesystem::exists(storage_path_)) {
            response->set_status(common::v1::HealthCheckResponse::SERVING_STATUS_NOT_SERVING);
            return grpc::Status::OK;
        }
        
        // Check if we can write to storage
        std::string test_file = storage_path_ + "/.health_check";
        std::ofstream test(test_file);
        if (test.is_open()) {
            test << "health_check";
            test.close();
            std::filesystem::remove(test_file);
        } else {
            response->set_status(common::v1::HealthCheckResponse::SERVING_STATUS_NOT_SERVING);
            return grpc::Status::OK;
        }
        
        response->set_status(common::v1::HealthCheckResponse::SERVING_STATUS_SERVING);
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        spdlog::error("💥 Health check failed: {}", e.what());
        response->set_status(common::v1::HealthCheckResponse::SERVING_STATUS_NOT_SERVING);
        return grpc::Status::OK;
    }
}

// Helper methods

std::string DICOMServiceImpl::generate_uuid() const {
    uuid_t uuid;
    uuid_generate(uuid);
    
    char uuid_str[37];
    uuid_unparse(uuid, uuid_str);
    
    return std::string(uuid_str);
}

bool DICOMServiceImpl::validate_dicom_compliance(const DcmFileFormat& fileformat) const {
    // Create a non-const copy for validation
    DcmFileFormat& non_const_fileformat = const_cast<DcmFileFormat&>(fileformat);
    DcmDataset* dataset = non_const_fileformat.getDataset();
    
    // Check required DICOM tags
    OFString sopClassUID;
    if (dataset->findAndGetOFString(DCM_SOPClassUID, sopClassUID).bad()) {
        spdlog::warn("⚠️ Missing SOP Class UID");
        return false;
    }
    
    OFString studyInstanceUID;
    if (dataset->findAndGetOFString(DCM_StudyInstanceUID, studyInstanceUID).bad()) {
        spdlog::warn("⚠️ Missing Study Instance UID");
        return false;
    }
    
    OFString seriesInstanceUID;
    if (dataset->findAndGetOFString(DCM_SeriesInstanceUID, seriesInstanceUID).bad()) {
        spdlog::warn("⚠️ Missing Series Instance UID");
        return false;
    }
    
    OFString sopInstanceUID;
    if (dataset->findAndGetOFString(DCM_SOPInstanceUID, sopInstanceUID).bad()) {
        spdlog::warn("⚠️ Missing SOP Instance UID");
        return false;
    }
    
    return true;
}

::dicom::v1::DICOMMetadata DICOMServiceImpl::extract_metadata(DcmDataset* dataset) const {
    ::dicom::v1::DICOMMetadata metadata;
    
    OFString value;
    
    if (dataset->findAndGetOFString(DCM_PatientID, value).good()) {
        metadata.set_patient_id(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_PatientName, value).good()) {
        metadata.set_patient_name(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_PatientBirthDate, value).good()) {
        metadata.set_patient_birth_date(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_PatientSex, value).good()) {
        metadata.set_patient_sex(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_StudyInstanceUID, value).good()) {
        metadata.set_study_instance_uid(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_SeriesInstanceUID, value).good()) {
        metadata.set_series_instance_uid(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_SOPInstanceUID, value).good()) {
        metadata.set_sop_instance_uid(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_StudyDate, value).good()) {
        metadata.set_study_date(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_SeriesDate, value).good()) {
        metadata.set_series_date(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_Modality, value).good()) {
        metadata.set_modality(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_Manufacturer, value).good()) {
        metadata.set_manufacturer(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_ManufacturerModelName, value).good()) {
        metadata.set_manufacturer_model_name(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_InstitutionName, value).good()) {
        metadata.set_institution_name(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_ReferringPhysicianName, value).good()) {
        metadata.set_referring_physician_name(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_PerformingPhysicianName, value).good()) {
        metadata.set_performing_physician_name(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_StudyDescription, value).good()) {
        metadata.add_study_description(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_SeriesDescription, value).good()) {
        metadata.add_series_description(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_BodyPartExamined, value).good()) {
        metadata.set_body_part_examined(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_ViewPosition, value).good()) {
        metadata.set_view_position(value.c_str());
    }
    
    if (dataset->findAndGetOFString(DCM_PhotometricInterpretation, value).good()) {
        metadata.set_photometric_interpretation(value.c_str());
    }
    
    // Image-related metadata
    Uint16 rows = 0;
    if (dataset->findAndGetUint16(DCM_Rows, rows).good()) {
        metadata.set_rows(rows);
    }
    
    Uint16 columns = 0;
    if (dataset->findAndGetUint16(DCM_Columns, columns).good()) {
        metadata.set_columns(columns);
    }
    
    Float64 pixelSpacing = 0.0;
    if (dataset->findAndGetFloat64(DCM_PixelSpacing, pixelSpacing).good()) {
        metadata.set_pixel_spacing(pixelSpacing);
    }
    
    Float64 sliceThickness = 0.0;
    if (dataset->findAndGetFloat64(DCM_SliceThickness, sliceThickness).good()) {
        metadata.set_slice_thickness(sliceThickness);
    }
    
    Float64 windowCenter = 0.0;
    if (dataset->findAndGetFloat64(DCM_WindowCenter, windowCenter).good()) {
        metadata.set_window_center(windowCenter);
    }
    
    Float64 windowWidth = 0.0;
    if (dataset->findAndGetFloat64(DCM_WindowWidth, windowWidth).good()) {
        metadata.set_window_width(windowWidth);
    }
    
    return metadata;
}

std::vector<::dicom::v1::StoreDICOMResponse> DICOMServiceImpl::query_files(
    const std::string& patient_id,
    const std::string& modality,
    const google::protobuf::Timestamp& date_from,
    const google::protobuf::Timestamp& date_to,
    int32_t page_size,
    const std::string& page_token
) const {
    std::vector<::dicom::v1::StoreDICOMResponse> query_results;
    
    try {
        for (const auto& study_entry : std::filesystem::directory_iterator(storage_path_)) {
            if (!study_entry.is_directory()) continue;
            
            for (const auto& file_entry : std::filesystem::directory_iterator(study_entry.path())) {
                if (!file_entry.is_regular_file() || file_entry.path().extension() != ".dcm") {
                    continue;
                }
                
                // Load DICOM file to check metadata
                DcmFileFormat fileformat;
                OFCondition status = fileformat.loadFile(file_entry.path().c_str());
                
                if (status.good()) {
                    DcmDataset* dataset = fileformat.getDataset();
                    auto metadata = extract_metadata(dataset);
                    
                    // Apply filters
                    if (!patient_id.empty() && metadata.patient_id() != patient_id) {
                        continue;
                    }
                    
                    if (!modality.empty() && metadata.modality() != modality) {
                        continue;
                    }
                    
                    // Create response
                    ::dicom::v1::StoreDICOMResponse response;
                    response.set_dicom_id(file_entry.path().stem());
                    response.set_patient_id(metadata.patient_id());
                    response.set_patient_name(metadata.patient_name());
                    response.set_study_date(metadata.study_date());
                    response.set_modality(metadata.modality());
                    response.set_file_path(file_entry.path());
                    response.set_file_size(std::filesystem::file_size(file_entry.path()));
                    response.set_study_instance_uid(metadata.study_instance_uid());
                    response.set_series_instance_uid(metadata.series_instance_uid());
                    response.set_sop_instance_uid(metadata.sop_instance_uid());
                    
                    query_results.push_back(response);
                }
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("💥 Error querying files: {}", e.what());
    }
    
    return query_results;
}

// Helper function for DICOM calibration
float apply_dicom_calibration(uint16_t raw_value, double sensitivity, double baseline) {
    // DICOM calibration formula: calibrated_value = (raw_value * sensitivity) + baseline
    // This converts raw ADC values to medical units (typically mV for ECG)
    return static_cast<float>(raw_value * sensitivity + baseline);
}

std::optional<::dicom::v1::WaveformData> DICOMServiceImpl::extract_ecg_waveform(DcmDataset* dataset) const {
    spdlog::info("🔍 Extracting ECG waveform from DICOM dataset");
    
    if (!dataset) {
        spdlog::error("❌ Null dataset provided to waveform extractor");
        return std::nullopt;
    }
    
    // Find waveform sequence
    DcmElement* element = nullptr;
    OFCondition status = dataset->findAndGetElement(DCM_WaveformSequence, element);
    
    if (status.bad() || !element) {
        spdlog::warn("⚠️ No waveform sequence found in DICOM dataset");
        return std::nullopt;
    }
    
    DcmSequenceOfItems* sequence = dynamic_cast<DcmSequenceOfItems*>(element);
    if (!sequence || sequence->card() == 0) {
        spdlog::warn("⚠️ Empty waveform sequence found");
        return std::nullopt;
    }
    
    spdlog::info("📊 Found waveform sequence with {} items", sequence->card());
    
    // Initialize waveform structure
    ::dicom::v1::WaveformData waveform;
    std::vector<std::vector<float>> channel_data;
    std::vector<std::string> global_channel_names;
    
    // Variables to store metadata from first item
    Uint16 totalChannels = 0;
    Uint32 totalSamples = 0;
    Float64 overallSamplingFrequency = 0.0;
    
    // Process each waveform item (typically one per channel group)
    for (size_t item_idx = 0; item_idx < sequence->card(); ++item_idx) {
        DcmItem* item = sequence->getItem(item_idx);
        if (!item) {
            spdlog::warn("⚠️ Could not get waveform item {}", item_idx);
            continue;
        }
        
        spdlog::debug("Processing waveform item {}", item_idx);
        
        // Extract waveform metadata
        Uint16 numChannels = 0;
        Uint32 numSamples = 0;
        Float64 samplingFrequency = 0.0;
        
        status = item->findAndGetUint16(DCM_NumberOfWaveformChannels, numChannels);
        if (status.bad()) {
            spdlog::warn("⚠️ Could not get number of channels for item {}", item_idx);
            continue;
        }
        
        status = item->findAndGetUint32(DCM_NumberOfWaveformSamples, numSamples);
        if (status.bad()) {
            spdlog::warn("⚠️ Could not get number of samples for item {}", item_idx);
            continue;
        }
        
        status = item->findAndGetFloat64(DCM_SamplingFrequency, samplingFrequency);
        if (status.bad()) {
            spdlog::warn("⚠️ Could not get sampling frequency for item {}", item_idx);
            samplingFrequency = 250.0; // Default ECG sampling rate
        }
        
        // Store metadata from first item
        if (item_idx == 0) {
            totalChannels = numChannels;
            totalSamples = numSamples;
            overallSamplingFrequency = samplingFrequency;
            
            // Set waveform metadata
            waveform.set_num_channels(totalChannels);
            waveform.set_num_samples(totalSamples);
            waveform.set_sampling_frequency(overallSamplingFrequency);
            
            // Initialize channel data vectors
            channel_data.resize(totalChannels);
            for (int ch = 0; ch < totalChannels; ++ch) {
                channel_data[ch].reserve(totalSamples);
            }
        }
        
        // Extract channel-specific calibration parameters
        std::vector<double> sensitivities(numChannels, 1.0);
        std::vector<double> baselines(numChannels, 0.0);
        std::vector<std::string> local_channel_names(numChannels);
        
        for (int ch = 0; ch < numChannels; ++ch) {
            // Get channel sensitivity
            DcmElement* sensElement = nullptr;
            if (item->findAndGetElement(DCM_ChannelSensitivity, sensElement).good() && sensElement) {
                DcmSequenceOfItems* sensSequence = dynamic_cast<DcmSequenceOfItems*>(sensElement);
                if (sensSequence && sensSequence->card() > ch) {
                    DcmItem* sensItem = sensSequence->getItem(ch);
                    if (sensItem) {  
                        Float64 sensitivity = 1.0;
                        if (sensItem->findAndGetFloat64(DCM_ChannelSensitivity, sensitivity).good()) {
                            sensitivities[ch] = sensitivity;
                        }
                    }
                }
            }
            
            // Get channel baseline
            DcmElement* baselineElement = nullptr;
            if (item->findAndGetElement(DCM_ChannelBaseline, baselineElement).good() && baselineElement) {
                DcmSequenceOfItems* baselineSequence = dynamic_cast<DcmSequenceOfItems*>(baselineElement);
                if (baselineSequence && baselineSequence->card() > ch) {
                    DcmItem* baselineItem = baselineSequence->getItem(ch);
                    if (baselineItem) {  // ✅ NULL CHECK EKLE!
                        Float64 baseline = 0.0;
                        if (baselineItem->findAndGetFloat64(DCM_ChannelBaseline, baseline).good()) {
                            baselines[ch] = baseline;
                        }
                    }
                }
            }
            
            // Get channel name
            DcmElement* labelElement = nullptr;
            if (item->findAndGetElement(DCM_ChannelLabel, labelElement).good() && labelElement) {
                DcmSequenceOfItems* labelSequence = dynamic_cast<DcmSequenceOfItems*>(labelElement);
                if (labelSequence && labelSequence->card() > ch) {
                    DcmItem* labelItem = labelSequence->getItem(ch);
                    if (labelItem) {  // ✅ NULL CHECK EKLE!
                        OFString label;
                        if (labelItem->findAndGetOFString(DCM_ChannelLabel, label).good()) {
                            local_channel_names[ch] = std::string(label.c_str());
                        } else {
                            local_channel_names[ch] = "Channel_" + std::to_string(ch + 1);
                        }
                    } else {
                        local_channel_names[ch] = "Channel_" + std::to_string(ch + 1);
                    }
                }
            } else {
                local_channel_names[ch] = "Channel_" + std::to_string(ch + 1);
            }
        }
        
        // Add local channel names to global list
        global_channel_names.insert(global_channel_names.end(), local_channel_names.begin(), local_channel_names.end());
        
        // Get waveform data
        const Uint16* waveformData = nullptr;
        unsigned long count = 0;
        status = item->findAndGetUint16Array(DCM_WaveformData, waveformData, &count);
        
        if (status.bad() || !waveformData) {
            spdlog::warn("⚠️ No waveform data found for item {}", item_idx);
            continue;
        }
        
        spdlog::info("📈 Extracting {} samples for {} channels", count, totalChannels);
        
        // Verify data size
        if (count != totalChannels * totalSamples) {
            spdlog::warn("⚠️ Data size mismatch: expected {}, got {}", 
                        totalChannels * totalSamples, count);
            continue;
        }
        
        // Extract and calibrate samples for each channel
        for (unsigned long sample_idx = 0; sample_idx < count; ++sample_idx) {
            int channel_idx = sample_idx % totalChannels;
            int sample_in_channel = sample_idx / totalChannels;
            
            if (sample_in_channel >= totalSamples) break;
            
            // Apply DICOM calibration: calibrated = (raw * sensitivity) + baseline
            float calibrated_value = apply_dicom_calibration(
                waveformData[sample_idx],
                sensitivities[channel_idx],
                baselines[channel_idx]
            );
            
            channel_data[channel_idx].push_back(calibrated_value);
        }
        
        // Store calibration info from first item
        if (item_idx == 0) {
            waveform.set_sensitivity(sensitivities[0]); // Use first channel as representative
            waveform.set_baseline(baselines[0]);
            waveform.set_units("mV"); // Standard ECG units
            
            // Extract bit depth information
            Uint16 bitsAllocated = 16;
            Uint16 bitsStored = 16;
            item->findAndGetUint16(DCM_WaveformBitsAllocated, bitsAllocated);
            item->findAndGetUint16(DCM_WaveformBitsStored, bitsStored);
            
            waveform.set_bits_allocated(bitsAllocated);
            waveform.set_bits_stored(bitsStored);
            waveform.set_is_signed(true); // Assume signed for ECG
            
            spdlog::debug("📊 Waveform metadata: {} channels, {} samples, {:.1f} Hz, {} bits", 
                        totalChannels, totalSamples, overallSamplingFrequency, bitsStored);
        }
    }
    
    // Check if we extracted any data
    if (channel_data.empty() || channel_data[0].empty()) {
        spdlog::warn("⚠️ No valid waveform data extracted");
        return std::nullopt;
    }
    
    // Add channel names to waveform
    for (const auto& name : global_channel_names) {
        waveform.add_channel_names(name);
    }
    
    // Interleave channel data (DICOM stores it this way)
    std::vector<float> interleaved_data;
    interleaved_data.reserve(totalChannels * totalSamples);
    
    for (int sample_idx = 0; sample_idx < totalSamples; ++sample_idx) {
        for (int ch = 0; ch < totalChannels; ++ch) {
            if (sample_idx < channel_data[ch].size()) {
                interleaved_data.push_back(channel_data[ch][sample_idx]);
            }
        }
    }
    
    // Set waveform data
    for (float sample : interleaved_data) {
        waveform.add_samples(sample);
    }
    
    spdlog::info("✅ Successfully extracted ECG waveform: {} channels, {} samples, {} data points", 
                totalChannels, totalSamples, interleaved_data.size());
    
    return waveform;
}

bool DICOMServiceImpl::is_file_accessible(const std::string& file_path) const {
    return std::filesystem::exists(file_path) && 
           std::filesystem::is_regular_file(file_path);
}

} // namespace dicom
} // namespace vitalstream
