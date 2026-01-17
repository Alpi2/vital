#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>
#include "../src/dicom_service_impl.h"
#include "../src/storage_manager.h"
#include "../src/audit_logger.h"
#include "dicom/v1/dicom_service.pb.h"
#include "common/v1/common.pb.h"

using namespace vitalstream::dicom;
using ::testing::_;
using ::testing::Return;
using ::testing::NiceMock;

class DICOMServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary test directory
        test_dir_ = std::filesystem::temp_directory_path() / ("dicom_test_" + std::to_string(std::time(nullptr)));
        std::filesystem::create_directories(test_dir_);
        
        // Initialize components
        storage_manager_ = std::make_shared<StorageManager>(test_dir_.string());
        audit_logger_ = std::make_shared<AuditLogger>((test_dir_ / "audit.log").string());
        service_ = std::make_unique<DICOMServiceImpl>(test_dir_.string(), audit_logger_);
    }
    
    void TearDown() override {
        // Clean up test directory
        std::filesystem::remove_all(test_dir_);
    }
    
    std::filesystem::path test_dir_;
    std::shared_ptr<StorageManager> storage_manager_;
    std::shared_ptr<AuditLogger> audit_logger_;
    std::unique_ptr<DICOMServiceImpl> service_;
};

// Test DICOM file storage
TEST_F(DICOMServiceTest, StoreDICOM_Success) {
    grpc::ServerContext context;
    ::dicom::v1::StoreDICOMRequest request;
    ::dicom::v1::StoreDICOMResponse response;
    
    // Create test DICOM data
    std::string test_dicom_data = "DICOM_TEST_DATA";
    request.set_file_data(test_dicom_data);
    request.set_patient_id("TEST_PATIENT_001");
    // Modality is extracted from DICOM data, not set in request
    
    auto start_time = std::chrono::high_resolution_clock::now();
    grpc::Status status = service_->StoreDICOM(&context, &request, &response);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    EXPECT_TRUE(status.ok());
    EXPECT_GT(response.dicom_id().length(), 0);
    EXPECT_EQ(response.patient_id(), "TEST_PATIENT_001");
    // Modality is extracted from DICOM data, not set in request
    EXPECT_LT(duration.count(), 100); // Should be < 100ms
    EXPECT_TRUE(std::filesystem::exists(test_dir_ / "dicom" / (response.dicom_id() + ".dcm")));
}

// Test DICOM file retrieval
TEST_F(DICOMServiceTest, GetDICOM_Success) {
    grpc::ServerContext context;
    
    // First store a file
    ::dicom::v1::StoreDICOMRequest store_request;
    ::dicom::v1::StoreDICOMResponse store_response;
    store_request.set_file_data("TEST_DICOM_DATA");
    store_request.set_patient_id("TEST_PATIENT_001");
    // Modality is extracted from DICOM data, not set in request
    
    grpc::Status store_status = service_->StoreDICOM(&context, &store_request, &store_response);
    ASSERT_TRUE(store_status.ok());
    
    // Now retrieve it
    ::dicom::v1::GetDICOMRequest get_request;
    ::dicom::v1::GetDICOMResponse get_response;
    get_request.set_dicom_id(store_response.dicom_id());
    
    auto start_time = std::chrono::high_resolution_clock::now();
    grpc::Status get_status = service_->GetDICOM(&context, &get_request, &get_response);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    EXPECT_TRUE(get_status.ok());
    EXPECT_EQ(get_response.file_data(), "TEST_DICOM_DATA");
    EXPECT_LT(duration.count(), 50); // Should be < 50ms
}

// Test DICOM file not found
TEST_F(DICOMServiceTest, GetDICOM_NotFound) {
    grpc::ServerContext context;
    ::dicom::v1::GetDICOMRequest request;
    ::dicom::v1::GetDICOMResponse response;
    request.set_dicom_id("NON_EXISTENT_ID");
    
    grpc::Status status = service_->GetDICOM(&context, &request, &response);
    
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.error_code(), grpc::StatusCode::NOT_FOUND);
}

// Test waveform extraction
TEST_F(DICOMServiceTest, ExtractWaveform_Success) {
    grpc::ServerContext context;
    
    // Create a mock DICOM file with waveform data
    ::dicom::v1::StoreDICOMRequest store_request;
    ::dicom::v1::StoreDICOMResponse store_response;
    
    // Create test DICOM data with waveform (simplified)
    std::string dicom_with_waveform = "DICOM_WITH_WAVEFORM_DATA";
    store_request.set_file_data(dicom_with_waveform);
    store_request.set_patient_id("TEST_PATIENT_001");
    // Modality is extracted from DICOM data, not set in request
    
    grpc::Status store_status = service_->StoreDICOM(&context, &store_request, &store_response);
    ASSERT_TRUE(store_status.ok());
    
    // Extract waveform
    ::dicom::v1::ExtractWaveformRequest extract_request;
    ::dicom::v1::ExtractWaveformResponse extract_response;
    extract_request.set_dicom_id(store_response.dicom_id());
    
    auto start_time = std::chrono::high_resolution_clock::now();
    grpc::Status extract_status = service_->ExtractWaveform(&context, &extract_request, &extract_response);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    EXPECT_TRUE(extract_status.ok());
    EXPECT_LT(duration.count(), 50); // Should be < 50ms
}

// Test DICOM query
TEST_F(DICOMServiceTest, QueryDICOM_Success) {
    grpc::ServerContext context;
    
    // First store some files
    for (int i = 0; i < 3; ++i) {
        ::dicom::v1::StoreDICOMRequest store_request;
        ::dicom::v1::StoreDICOMResponse store_response;
        store_request.set_file_data("DICOM_DATA_" + std::to_string(i));
        store_request.set_patient_id("TEST_PATIENT_001");
        // Modality is extracted from DICOM data, not set in request
        
        grpc::Status status = service_->StoreDICOM(&context, &store_request, &store_response);
        ASSERT_TRUE(status.ok());
    }
    
    // Query files
    ::dicom::v1::QueryDICOMRequest query_request;
    ::dicom::v1::QueryDICOMResponse query_response;
    query_request.set_patient_id("TEST_PATIENT_001");
    // Modality is extracted from DICOM data, not set in request
    query_request.set_page_size(10);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    grpc::Status query_status = service_->QueryDICOM(&context, &query_request, &query_response);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    EXPECT_TRUE(query_status.ok());
    EXPECT_EQ(query_response.dicom_files_size(), 3);
    EXPECT_LT(duration.count(), 100); // Should be < 100ms
}

// Performance test
TEST_F(DICOMServiceTest, Performance_StoreDICOM) {
    const int num_files = 100;
    std::vector<std::chrono::milliseconds> durations;
    
    for (int i = 0; i < num_files; ++i) {
        grpc::ServerContext context;
        ::dicom::v1::StoreDICOMRequest request;
        ::dicom::v1::StoreDICOMResponse response;
        
        request.set_file_data("PERF_TEST_DATA_" + std::to_string(i));
        request.set_patient_id("PERF_PATIENT_" + std::to_string(i % 10));
        // Modality is extracted from DICOM data, not set in request
        
        auto start_time = std::chrono::high_resolution_clock::now();
        grpc::Status status = service_->StoreDICOM(&context, &request, &response);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        ASSERT_TRUE(status.ok());
        durations.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
    
    // Calculate statistics
    long long total_time = 0;
    long long max_time = 0;
    long long min_time = LLONG_MAX;
    
    for (const auto& duration : durations) {
        total_time += duration.count();
        max_time = std::max(max_time, duration.count());
        min_time = std::min(min_time, duration.count());
    }
    
    double avg_time = static_cast<double>(total_time) / num_files;
    
    EXPECT_LT(avg_time, 10.0); // Average should be < 10ms
    EXPECT_LT(max_time, 50.0); // Max should be < 50ms
    
    std::cout << "Performance Test Results:" << std::endl;
    std::cout << "  Average: " << avg_time << "ms" << std::endl;
    std::cout << "  Min: " << min_time << "ms" << std::endl;
    std::cout << "  Max: " << max_time << "ms" << std::endl;
    std::cout << "  Total: " << total_time << "ms for " << num_files << " files" << std::endl;
}

// Test error handling
TEST_F(DICOMServiceTest, ErrorHandling_EmptyRequest) {
    grpc::ServerContext context;
    ::dicom::v1::StoreDICOMRequest request;
    ::dicom::v1::StoreDICOMResponse response;
    
    // Empty request should fail
    grpc::Status status = service_->StoreDICOM(&context, &request, &response);
    
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

// Test audit logging
TEST_F(DICOMServiceTest, AuditLogging_Enabled) {
    grpc::ServerContext context;
    ::dicom::v1::StoreDICOMRequest request;
    ::dicom::v1::StoreDICOMResponse response;
    
    request.set_file_data("AUDIT_TEST_DATA");
    request.set_patient_id("AUDIT_PATIENT");
    // Modality is extracted from DICOM data, not set in request
    
    grpc::Status status = service_->StoreDICOM(&context, &request, &response);
    
    EXPECT_TRUE(status.ok());
    
    // Check if audit log was created
    EXPECT_TRUE(std::filesystem::exists(test_dir_ / "audit.log"));
    
    // Check audit log content
    std::ifstream audit_file(test_dir_ / "audit.log");
    std::string content((std::istreambuf_iterator<char>(audit_file)),
                        std::istreambuf_iterator<char>());
    
    EXPECT_TRUE(content.find("AUDIT") != std::string::npos);
    EXPECT_TRUE(content.find("AUDIT_PATIENT") != std::string::npos);
}
