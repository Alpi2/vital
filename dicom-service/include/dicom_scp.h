/**
 * @file dicom_scp.h
 * @brief DICOM Service Class Provider (SCP) for VitalStream
 * 
 * Implements DICOM network services:
 * - C-STORE (storage)
 * - C-FIND (query)
 * - C-MOVE (retrieve)
 * - C-ECHO (verification)
 * 
 * @author VitalStream Development Team
 * @date 2026-01-03
 * @version 1.0.0
 */

#ifndef VITALSTREAM_DICOM_SCP_H
#define VITALSTREAM_DICOM_SCP_H

#ifdef HAVE_DCMTK
#include <dcmtk/dcmnet/assoc.h>
#include <dcmtk/dcmnet/dimse.h>
#include <dcmtk/dcmdata/dcdatset.h>
#else
// Minimal fallbacks so file compiles without DCMTK
using OFCondition = int;
class DcmDataset;
struct T_ASC_Association;
struct T_DIMSE_Message;
struct T_ASC_Network {}; // minimal placeholder
#endif

#include <string>
#include <memory>
#include <functional>
#include <map>
#include <thread>
#include <mutex>

namespace vitalstream {
namespace dicom {

/**
 * @brief DICOM SCP configuration
 */
struct SCPConfig {
    std::string ae_title = "VITALSTREAM_SCP";  ///< Application Entity Title
    int port = 11112;                           ///< DICOM port
    int max_pdu_size = 16384;                   ///< Maximum PDU size (bytes)
    int timeout = 30;                           ///< Network timeout (seconds)
    std::string storage_path = "/var/dicom";   ///< Storage directory
    bool enable_compression = true;             ///< Enable JPEG compression
    bool validate_incoming = true;              ///< Validate incoming datasets
};

/**
 * @brief DICOM association information
 */
struct AssociationInfo {
    std::string calling_ae;     ///< Calling AE Title
    std::string called_ae;      ///< Called AE Title
    std::string peer_ip;        ///< Peer IP address
    int peer_port;              ///< Peer port
    time_t start_time;          ///< Association start time
};

/**
 * @brief Callback for C-STORE operations
 * @param dataset Received DICOM dataset
 * @param assoc_info Association information
 * @return Status code (0 = success)
 */
using StoreCallback = std::function<OFCondition(
    DcmDataset* dataset,
    const AssociationInfo& assoc_info
)>;

/**
 * @brief Callback for C-FIND operations
 * @param query Query dataset
 * @param assoc_info Association information
 * @return List of matching datasets
 */
using FindCallback = std::function<std::vector<DcmDataset*>(
    DcmDataset* query,
    const AssociationInfo& assoc_info
)>;

/**
 * @brief DICOM Service Class Provider
 * 
 * Implements DICOM network services as an SCP (server).
 * Supports C-STORE, C-FIND, C-MOVE, and C-ECHO.
 */
class DicomSCP {
public:
    /**
     * @brief Constructor
     * @param config SCP configuration
     */
    explicit DicomSCP(const SCPConfig& config = SCPConfig());
    
    /**
     * @brief Destructor
     */
    ~DicomSCP();

    // Disable copy
    DicomSCP(const DicomSCP&) = delete;
    DicomSCP& operator=(const DicomSCP&) = delete;

    /**
     * @brief Start the SCP server
     * @return true if started successfully
     */
    bool start();

    /**
     * @brief Stop the SCP server
     */
    void stop();

    /**
     * @brief Check if server is running
     * @return true if running
     */
    bool isRunning() const { return running_; }

    /**
     * @brief Register C-STORE callback
     * @param callback Callback function
     */
    void registerStoreCallback(StoreCallback callback);

    /**
     * @brief Register C-FIND callback
     * @param callback Callback function
     */
    void registerFindCallback(FindCallback callback);

    /**
     * @brief Get configuration
     * @return Current configuration
     */
    const SCPConfig& getConfig() const { return config_; }

    /**
     * @brief Get active associations count
     * @return Number of active associations
     */
    size_t getActiveAssociations() const;

    /**
     * @brief Get statistics
     * @return Statistics map (key: metric name, value: count)
     */
    std::map<std::string, uint64_t> getStatistics() const;

private:
    /**
     * @brief Accept association
     * @param assoc Association handle
     * @return Status code
     */
    OFCondition acceptAssociation(T_ASC_Association** assoc);

    /**
     * @brief Handle C-ECHO request
     * @param assoc Association
     * @param msg DIMSE message
     * @return Status code
     */
    OFCondition handleEchoRequest(
        T_ASC_Association* assoc,
        T_DIMSE_Message* msg
    );

    /**
     * @brief Handle C-STORE request
     * @param assoc Association
     * @param msg DIMSE message
     * @return Status code
     */
    OFCondition handleStoreRequest(
        T_ASC_Association* assoc,
        T_DIMSE_Message* msg
    );

    /**
     * @brief Handle C-FIND request
     * @param assoc Association
     * @param msg DIMSE message
     * @return Status code
     */
    OFCondition handleFindRequest(
        T_ASC_Association* assoc,
        T_DIMSE_Message* msg
    );

    /**
     * @brief Handle C-MOVE request
     * @param assoc Association
     * @param msg DIMSE message
     * @return Status code
     */
    OFCondition handleMoveRequest(
        T_ASC_Association* assoc,
        T_DIMSE_Message* msg
    );

    /**
     * @brief Network thread function
     */
    void networkThread();

    SCPConfig config_;                  ///< Configuration
    T_ASC_Network* network_ = nullptr;  ///< DICOM network
    bool running_ = false;              ///< Running flag
    std::unique_ptr<std::thread> thread_; ///< Network thread
    
    StoreCallback store_callback_;      ///< C-STORE callback
    FindCallback find_callback_;        ///< C-FIND callback
    
    // Statistics
    mutable std::mutex stats_mutex_;
    uint64_t total_associations_ = 0;
    uint64_t total_stores_ = 0;
    uint64_t total_finds_ = 0;
    uint64_t total_moves_ = 0;
    uint64_t total_echos_ = 0;
    uint64_t failed_operations_ = 0;
};

} // namespace dicom
} // namespace vitalstream

#endif // VITALSTREAM_DICOM_SCP_H
