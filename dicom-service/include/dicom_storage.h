#ifndef VITALSTREAM_DICOM_STORAGE_H
#define VITALSTREAM_DICOM_STORAGE_H

#include <string>

namespace vitalstream {
namespace dicom {

bool storeDicomFile(const std::string& filepath);

} // namespace dicom
} // namespace vitalstream

#endif // VITALSTREAM_DICOM_STORAGE_H
