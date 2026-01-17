#ifndef VITALSTREAM_DICOM_QUERY_H
#define VITALSTREAM_DICOM_QUERY_H

#include <vector>
#include <string>

namespace vitalstream {
namespace dicom {

std::vector<std::string> findDicomByPatientId(const std::string& patientId);

} // namespace dicom
} // namespace vitalstream

#endif // VITALSTREAM_DICOM_QUERY_H
