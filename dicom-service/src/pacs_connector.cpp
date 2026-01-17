#include "dicom_scp.h"
#include <iostream>

namespace vitalstream {
namespace dicom {

bool sendToPACS(const std::string& ae, const std::string& operation, const std::string& payload) {
    std::cout << "sendToPACS stub: ae=" << ae << " op=" << operation << std::endl;
    return true;
}

} // namespace dicom
} // namespace vitalstream
