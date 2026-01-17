#include "dicom_scp.h"
#include <iostream>
#include <thread>

namespace vitalstream {
namespace dicom {

DicomSCP::DicomSCP(const SCPConfig& config) : config_(config) {}

DicomSCP::~DicomSCP() {
    stop();
}

bool DicomSCP::start() {
#ifdef NO_DCMTK
    // Minimal stub behavior when DCMTK is not available
    std::cout << "DicomSCP: running in stub mode (no DCMTK)" << std::endl;
    running_ = true;
    // Simulate a short running period
    return true;
#else
    // TODO: real implementation using DCMTK
    running_ = true;
    return true;
#endif
}

void DicomSCP::stop() {
    running_ = false;
}

size_t DicomSCP::getActiveAssociations() const {
    return total_associations_;
}

void DicomSCP::registerStoreCallback(StoreCallback callback) {
    store_callback_ = callback;
}

void DicomSCP::registerFindCallback(FindCallback callback) {
    find_callback_ = callback;
}

std::map<std::string, uint64_t> DicomSCP::getStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return {
        {"total_associations", total_associations_},
        {"total_stores", total_stores_},
        {"total_finds", total_finds_},
        {"total_moves", total_moves_},
        {"total_echos", total_echos_},
        {"failed_operations", failed_operations_}
    };
}

} // namespace dicom
} // namespace vitalstream
