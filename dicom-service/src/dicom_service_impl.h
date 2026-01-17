#pragma once

#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>
#include <filesystem>
#include <mutex>
#include "dicom/v1/dicom_service.grpc.pb.h"
#include "audit_logger.h"
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdatset.h>
#include <dcmtk/dcmdata/dcitem.h>

namespace vitalstream {
namespace dicom {

class DICOMServiceImpl final : public ::dicom::v1::DICOMService::Service {
public:
    explicit DICOMServiceImpl(const std::string& storage_path, 
                             std::shared_ptr<AuditLogger> audit_logger);
    
    // gRPC service implementations
    grpc::Status StoreDICOM(
        grpc::ServerContext* context,
        const ::dicom::v1::StoreDICOMRequest* request,
        ::dicom::v1::StoreDICOMResponse* response
    ) override;
    
    grpc::Status GetDICOM(
        grpc::ServerContext* context,
        const ::dicom::v1::GetDICOMRequest* request,
        ::dicom::v1::GetDICOMResponse* response
    ) override;
    
    grpc::Status ExtractWaveform(
        grpc::ServerContext* context,
        const ::dicom::v1::ExtractWaveformRequest* request,
        ::dicom::v1::ExtractWaveformResponse* response
    ) override;
    
    grpc::Status QueryDICOM(
        grpc::ServerContext* context,
        const ::dicom::v1::QueryDICOMRequest* request,
        ::dicom::v1::QueryDICOMResponse* response
    ) override;
    
    grpc::Status DeleteDICOM(
        grpc::ServerContext* context,
        const ::dicom::v1::DeleteDICOMRequest* request,
        ::dicom::v1::DeleteDICOMResponse* response
    ) override;
    
    grpc::Status ListDICOM(
        grpc::ServerContext* context,
        const ::dicom::v1::ListDICOMRequest* request,
        ::dicom::v1::ListDICOMResponse* response
    ) override;
    
    grpc::Status HealthCheck(
        grpc::ServerContext* context,
        const common::v1::HealthCheckRequest* request,
        common::v1::HealthCheckResponse* response
    ) override;

private:
    std::string storage_path_;
    std::shared_ptr<AuditLogger> audit_logger_;
    mutable std::mutex mutex_;
    
    // Helper methods
    std::string generate_uuid() const;
    bool validate_dicom_compliance(const DcmFileFormat& fileformat) const;
    std::optional<::dicom::v1::WaveformData> extract_ecg_waveform(DcmDataset* dataset) const;
    bool is_file_accessible(const std::string& file_path) const;
    ::dicom::v1::DICOMMetadata extract_metadata(DcmDataset* dataset) const;
    std::vector<::dicom::v1::StoreDICOMResponse> query_files(
        const std::string& patient_id,
        const std::string& modality,
        const google::protobuf::Timestamp& date_from,
        const google::protobuf::Timestamp& date_to,
        int32_t page_size,
        const std::string& page_token
    ) const;
};

} // namespace dicom
} // namespace vitalstream
