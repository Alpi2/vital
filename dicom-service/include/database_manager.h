#ifndef VITALSTREAM_DATABASE_MANAGER_H
#define VITALSTREAM_DATABASE_MANAGER_H

#include <string>

namespace vitalstream {
namespace dicom {

class DatabaseManager {
public:
    DatabaseManager();
    bool connect(const std::string& conn);
};

} // namespace dicom
} // namespace vitalstream

#endif // VITALSTREAM_DATABASE_MANAGER_H
