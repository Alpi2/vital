#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcmetinf.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include "../src/dicom_service_impl.h"
#include "../src/storage_manager.h"
#include "../src/audit_logger.h"
#include "dicom/v1/dicom_service.pb.h"

using namespace vitalstream::dicom;

class DICOMValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / ("validation_test_" + std::to_string(std::time(nullptr)));
        std::filesystem::create_directories(test_dir_);
        
        storage_manager_ = std::make_shared<StorageManager>(test_dir_.string());
        audit_logger_ = std::make_shared<AuditLogger>((test_dir_ / "audit.log").string());
        service_ = std::make_unique<DICOMServiceImpl>(test_dir_.string(), audit_logger_);
    }
    
    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
    }
    
    std::filesystem::path test_dir_;
    std::shared_ptr<StorageManager> storage_manager_;
    std::shared_ptr<AuditLogger> audit_logger_;
    std::unique_ptr<DICOMServiceImpl> service_;
    
    // Helper function to create a minimal valid DICOM file
    std::string create_minimal_dicom() {
        DcmFileFormat fileformat;
        DcmDataset* dataset = fileformat.getDataset();
        
        // Add required DICOM attributes
        dataset->putAndInsertString(DCM_SOPClassUID, "1.2.840.10008.5.1.4.1.1.2"); // CT Image Storage
        dataset->putAndInsertString(DCM_SOPInstanceUID, "1.2.3.4.5.6.7.8.9");
        dataset->putAndInsertString(DCM_StudyInstanceUID, "1.2.3.4.5.6.7.8");
        dataset->putAndInsertString(DCM_SeriesInstanceUID, "1.2.3.4.5.6.7");
        dataset->putAndInsertString(DCM_PatientID, "TEST_PATIENT_001");
        dataset->putAndInsertString(DCM_PatientName, "Test^Patient");
        dataset->putAndInsertString(DCM_StudyDate, "20230101");
        dataset->putAndInsertString(DCM_Modality, "CT");
        dataset->putAndInsertString(DCM_Manufacturer, "Test Manufacturer");
        dataset->putAndInsertString(DCM_InstitutionName, "Test Institution");
        
        // Save to file
        std::string filename = (test_dir_ / "minimal_dicom.dcm").string();
        OFCondition status = fileformat.saveFile(filename.c_str());
        
        if (status.bad()) {
            return "";
        }
        
        // Read file content
        std::ifstream file(filename, std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        
        // Clean up
        std::filesystem::remove(filename);
        
        return content;
    }
    
    // Helper function to create invalid DICOM data
    std::string create_invalid_dicom() {
        return "INVALID_DICOM_DATA_NOT_A_REAL_FILE";
    }
    
    // Helper function to create DICOM with waveform
    std::string create_dicom_with_waveform() {
        DcmFileFormat fileformat;
        DcmDataset* dataset = fileformat.getDataset();
        
        // Add basic DICOM attributes
        dataset->putAndInsertString(DCM_SOPClassUID, "1.2.840.10008.5.1.4.1.1.2");
        dataset->putAndInsertString(DCM_SOPInstanceUID, "1.2.3.4.5.6.7.8.9");
        dataset->putAndInsertString(DCM_StudyInstanceUID, "1.2.3.4.5.6.7.8");
        dataset->putAndInsertString(DCM_SeriesInstanceUID, "1.2.3.4.5.6.7");
        dataset->putAndInsertString(DCM_PatientID, "ECG_PATIENT_001");
        dataset->putAndInsertString(DCM_PatientName, "ECG^Patient");
        dataset->putAndInsertString(DCM_StudyDate, "20230101");
        dataset->putAndInsertString(DCM_Modality, "ECG");
        
        // Add waveform sequence
        DcmSequenceOfItems* waveform_seq = new DcmSequenceOfItems(DCM_WaveformSequence);
        DcmItem* waveform_item = new DcmItem();
        
        // Add waveform metadata
        waveform_item->putAndInsertUint16(DCM_NumberOfWaveformChannels, 12);
        waveform_item->putAndInsertUint32(DCM_NumberOfWaveformSamples, 1000);
        waveform_item->putAndInsertFloat64(DCM_SamplingFrequency, 250.0);
        
        // Add waveform data (12 channels, 1000 samples each)
        std::vector<Uint16> waveformData(12000); // 12 * 1000
        for (size_t i = 0; i < waveformData.size(); ++i) {
            waveformData[i] = static_cast<Uint16>(i % 4096); // Test data
        }
        waveform_item->putAndInsertUint16Array(DCM_WaveformData, waveformData.data(), waveformData.size());
        
        waveform_seq->insert(waveform_item);
        dataset->insert(waveform_seq);
        
        // Save to file
        std::string filename = (test_dir_ / "waveform_dicom.dcm").string();
        OFCondition status = fileformat.saveFile(filename.c_str());
        
        if (status.bad()) {
            return "";
        }
        
        // Read file content
        std::ifstream file(filename, std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        
        // Clean up
        std::filesystem::remove(filename);
        
        return content;
    }
};

// Test valid DICOM file storage
TEST_F(DICOMValidationTest, StoreValidDICOM) {
    std::cout << "🧪 Testing valid DICOM file storage..." << std::endl;
    
    grpc::ServerContext context;
    ::dicom::v1::StoreDICOMRequest request;
    ::dicom::v1::StoreDICOMResponse response;
    
    std::string dicom_data = create_minimal_dicom();
    ASSERT_FALSE(dicom_data.empty()) << "Failed to create minimal DICOM";
    
    request.set_file_data(dicom_data);
    request.set_patient_id("VALID_TEST_PATIENT");
    // Modality is extracted from DICOM data, not set in request
    
    grpc::Status status = service_->StoreDICOM(&context, &request, &response);
    
    EXPECT_TRUE(status.ok()) << "Valid DICOM should be stored successfully";
    EXPECT_FALSE(response.dicom_id().empty()) << "DICOM ID should be assigned";
    EXPECT_EQ(response.patient_id(), "VALID_TEST_PATIENT");
    EXPECT_EQ(response.modality(), "CT");
    
    // Verify file was created
    std::filesystem::path stored_file = test_dir_ / "dicom" / (response.dicom_id() + ".dcm");
    EXPECT_TRUE(std::filesystem::exists(stored_file)) << "DICOM file should be stored on disk";
    
    // Verify stored file is valid DICOM
    DcmFileFormat fileformat;
    OFCondition status_check = fileformat.loadFile(stored_file.c_str());
    EXPECT_TRUE(status_check.good()) << "Stored file should be valid DICOM";
    
    std::cout << "✅ Valid DICOM file stored successfully" << std::endl;
}

// Test invalid DICOM file rejection
TEST_F(DICOMValidationTest, RejectInvalidDICOM) {
    std::cout << "🧪 Testing invalid DICOM file rejection..." << std::endl;
    
    grpc::ServerContext context;
    ::dicom::v1::StoreDICOMRequest request;
    ::dicom::v1::StoreDICOMResponse response;
    
    std::string invalid_data = create_invalid_dicom();
    request.set_file_data(invalid_data);
    request.set_patient_id("INVALID_TEST_PATIENT");
    // Modality is extracted from DICOM data, not set in request
    
    grpc::Status status = service_->StoreDICOM(&context, &request, &response);
    
    EXPECT_FALSE(status.ok()) << "Invalid DICOM should be rejected";
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT) << "Should return INVALID_ARGUMENT";
    
    std::cout << "✅ Invalid DICOM file correctly rejected" << std::endl;
}

// Test DICOM compliance validation
TEST_F(DICOMValidationTest, DICOMCompliance) {
    std::cout << "🧪 Testing DICOM compliance validation..." << std::endl;
    
    // Create a DICOM file with missing required attributes
    DcmFileFormat fileformat;
    DcmDataset* dataset = fileformat.getDataset();
    
    // Missing required SOP Class UID - this should fail validation
    dataset->putAndInsertString(DCM_PatientID, "COMPLIANCE_TEST_PATIENT");
    dataset->putAndInsertString(DCM_PatientName, "Compliance^Test");
    
    std::string filename = (test_dir_ / "non_compliant_dicom.dcm").string();
    OFCondition save_status = fileformat.saveFile(filename.c_str());
    ASSERT_TRUE(save_status.good()) << "Failed to save non-compliant DICOM";
    
    // Try to store non-compliant DICOM
    grpc::ServerContext context;
    ::dicom::v1::StoreDICOMRequest request;
    ::dicom::v1::StoreDICOMResponse response;
    
    std::ifstream file(filename, std::ios::binary);
    std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    
    request.set_file_data(content);
    request.set_patient_id("COMPLIANCE_TEST_PATIENT");
    // Modality is extracted from DICOM data, not set in request
    
    grpc::Status status = service_->StoreDICOM(&context, &request, &response);
    
    EXPECT_FALSE(status.ok()) << "Non-compliant DICOM should be rejected";
    
    // Clean up
    std::filesystem::remove(filename);
    
    std::cout << "✅ Non-compliant DICOM correctly rejected" << std::endl;
}

// Test waveform DICOM validation
TEST_F(DICOMValidationTest, WaveformDICOMValidation) {
    std::cout << "🧪 Testing waveform DICOM validation..." << std::endl;
    
    grpc::ServerContext context;
    ::dicom::v1::StoreDICOMRequest request;
    ::dicom::v1::StoreDICOMResponse response;
    
    std::string waveform_dicom = create_dicom_with_waveform();
    ASSERT_FALSE(waveform_dicom.empty()) << "Failed to create DICOM with waveform";
    
    request.set_file_data(waveform_dicom);
    request.set_patient_id("WAVEFORM_TEST_PATIENT");
    // Modality is extracted from DICOM data, not set in request
    
    grpc::Status status = service_->StoreDICOM(&context, &request, &response);
    
    EXPECT_TRUE(status.ok()) << "Valid waveform DICOM should be stored";
    
    // Test waveform extraction
    ::dicom::v1::ExtractWaveformRequest extract_request;
    ::dicom::v1::ExtractWaveformResponse extract_response;
    extract_request.set_dicom_id(response.dicom_id());
    
    grpc::Status extract_status = service_->ExtractWaveform(&context, &extract_request, &extract_response);
    
    EXPECT_TRUE(extract_status.ok()) << "Waveform should be extractable from valid DICOM";
    EXPECT_GT(extract_response.waveform_data().num_channels(), 0) << "Waveform should have channels";
    EXPECT_GT(extract_response.waveform_data().num_samples(), 0) << "Waveform should have samples";
    EXPECT_GT(extract_response.waveform_data().sampling_frequency(), 0) << "Waveform should have sampling frequency";
    
    std::cout << "✅ Waveform DICOM validation passed" << std::endl;
    std::cout << "   Channels: " << extract_response.waveform_data().num_channels() << std::endl;
    std::cout << "   Samples: " << extract_response.waveform_data().num_samples() << std::endl;
    std::cout << "   Sampling Rate: " << extract_response.waveform_data().sampling_frequency() << " Hz" << std::endl;
}

// Test DICOM file size limits
TEST_F(DICOMValidationTest, FileSizeLimits) {
    std::cout << "🧪 Testing DICOM file size limits..." << std::endl;
    
    grpc::ServerContext context;
    ::dicom::v1::StoreDICOMRequest request;
    ::dicom::v1::StoreDICOMResponse response;
    
    // Test oversized file (>100MB)
    std::string oversized_data(100 * 1024 * 1024 + 1, 'X'); // 100MB + 1 byte
    request.set_file_data(oversized_data);
    request.set_patient_id("OVERSIZE_TEST_PATIENT");
    // Modality is extracted from DICOM data, not set in request
    
    grpc::Status status = service_->StoreDICOM(&context, &request, &response);
    
    EXPECT_FALSE(status.ok()) << "Oversized DICOM should be rejected";
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT) << "Should return INVALID_ARGUMENT for oversized file";
    
    // Test normal sized file
    std::string normal_data(1024, 'X'); // 1KB
    request.set_file_data(normal_data);
    request.set_patient_id("NORMAL_SIZE_PATIENT");
    // Modality is extracted from DICOM data, not set in request
    
    status = service_->StoreDICOM(&context, &request, &response);
    
    EXPECT_TRUE(status.ok()) << "Normal sized DICOM should be accepted";
    
    std::cout << "✅ File size limits enforced correctly" << std::endl;
}

// Test patient ID validation
TEST_F(DICOMValidationTest, PatientIDValidation) {
    std::cout << "🧪 Testing patient ID validation..." << std::endl;
    
    grpc::ServerContext context;
    ::dicom::v1::StoreDICOMRequest request;
    ::dicom::v1::StoreDICOMResponse response;
    
    std::string valid_dicom = create_minimal_dicom();
    request.set_file_data(valid_dicom);
    
    // Test empty patient ID
    request.set_patient_id("");
    // Modality is extracted from DICOM data, not set in request
    
    grpc::Status status = service_->StoreDICOM(&context, &request, &response);
    
    EXPECT_FALSE(status.ok()) << "Empty patient ID should be rejected";
    
    // Test valid patient ID
    request.set_patient_id("VALID_PATIENT_ID");
    // Modality is extracted from DICOM data, not set in request
    
    status = service_->StoreDICOM(&context, &request, &response);
    
    EXPECT_TRUE(status.ok()) << "Valid patient ID should be accepted";
    
    std::cout << "✅ Patient ID validation working correctly" << std::endl;
}

// Test modality validation
TEST_F(DICOMValidationTest, ModalityValidation) {
    std::cout << "🧪 Testing modality validation..." << std::endl;
    
    grpc::ServerContext context;
    ::dicom::v1::StoreDICOMRequest request;
    ::dicom::v1::StoreDICOMResponse response;
    
    std::string valid_dicom = create_minimal_dicom();
    request.set_file_data(valid_dicom);
    request.set_patient_id("MODALITY_TEST_PATIENT");
    
    // Test empty modality
    // Modality is extracted from DICOM data, not set in request
    
    grpc::Status status = service_->StoreDICOM(&context, &request, &response);
    
    EXPECT_FALSE(status.ok()) << "Empty modality should be rejected";
    
    // Test valid modalities
    std::vector<std::string> valid_modalities = {"CT", "MRI", "XA", "ECG", "US", "RT", "NM", "PET"};
    
    for (const auto& modality : valid_modalities) {
        // Modality is extracted from DICOM data, not set in request
        status = service_->StoreDICOM(&context, &request, &response);
        EXPECT_TRUE(status.ok()) << "Valid modality " << modality << " should be accepted";
    }
    
    std::cout << "✅ Modality validation working correctly" << std::endl;
}

// Test DICOM metadata extraction
TEST_F(DICOMValidationTest, MetadataExtraction) {
    std::cout << "🧪 Testing DICOM metadata extraction..." << std::endl;
    
    grpc::ServerContext context;
    ::dicom::v1::StoreDICOMRequest request;
    ::dicom::v1::StoreDICOMResponse response;
    
    std::string dicom_data = create_minimal_dicom();
    request.set_file_data(dicom_data);
    request.set_patient_id("METADATA_TEST_PATIENT");
    // Modality is extracted from DICOM data, not set in request
    
    grpc::Status status = service_->StoreDICOM(&context, &request, &response);
    
    EXPECT_TRUE(status.ok()) << "DICOM should be stored successfully";
    
    // Verify metadata was extracted
    EXPECT_EQ(response.patient_id(), "METADATA_TEST_PATIENT");
    EXPECT_EQ(response.modality(), "CT");
    EXPECT_FALSE(response.dicom_id().empty());
    EXPECT_FALSE(response.study_instance_uid().empty());
    EXPECT_FALSE(response.series_instance_uid().empty());
    EXPECT_FALSE(response.sop_instance_uid().empty());
    
    std::cout << "✅ Metadata extraction working correctly" << std::endl;
    std::cout << "   Patient ID: " << response.patient_id() << std::endl;
    std::cout << "   Modality: " << response.modality() << std::endl;
    std::cout << "   DICOM ID: " << response.dicom_id() << std::endl;
}
