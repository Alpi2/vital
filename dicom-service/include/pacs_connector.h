#ifndef VITALSTREAM_PACS_CONNECTOR_H
#define VITALSTREAM_PACS_CONNECTOR_H

#include <string>

namespace vitalstream {
namespace dicom {

bool sendToPACS(const std::string& ae, const std::string& operation, const std::string& payload);

} // namespace dicom
} // namespace vitalstream

#endif // VITALSTREAM_PACS_CONNECTOR_H
