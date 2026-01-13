#ifndef PATIENT_MANAGER_H
#define PATIENT_MANAGER_H

#include <QObject>
#include <QString>
#include <QDateTime>
#include <QMap>
#include <QVector>
#include <QJsonObject>
#include <memory>

namespace VitalStream {

/**
 * @brief Patient information structure
 */
struct PatientInfo {
    QString patientID;
    QString name;
    QString mrn;  // Medical Record Number
    QDateTime dateOfBirth;
    QString gender;
    QString bloodType;
    double height;  // cm
    double weight;  // kg
    QString allergies;
    QString diagnosis;
    QString admissionDate;
    QString dischargeDate;
    
    // Location
    QString department;
    QString floor;
    QString room;
    QString bed;
    
    // Status
    enum Status {
        Active,
        Discharged,
        Transferred,
        Deceased
    };
    Status status;
    
    // Priority (based on alarm severity)
    enum Priority {
        Critical,   // Red
        High,       // Orange
        Medium,     // Yellow
        Low,        // Green
        Normal      // White
    };
    Priority priority;
    
    // Monitoring
    bool isMonitored;
    QDateTime monitoringStartTime;
    
    QJsonObject toJson() const;
    static PatientInfo fromJson(const QJsonObject& json);
};

/**
 * @brief Bed information structure
 */
struct BedInfo {
    QString bedID;
    QString department;
    QString floor;
    QString room;
    QString bedNumber;
    bool isOccupied;
    QString patientID;  // If occupied
    bool hasMonitor;
    QString monitorID;
    
    enum BedType {
        ICU,
        StepDown,
        MedSurg,
        Emergency,
        Telemetry
    };
    BedType bedType;
    
    QJsonObject toJson() const;
    static BedInfo fromJson(const QJsonObject& json);
};

/**
 * @brief Patient transfer record
 */
struct TransferRecord {
    QString transferID;
    QString patientID;
    QString fromDepartment;
    QString fromBed;
    QString toDepartment;
    QString toBed;
    QDateTime transferTime;
    QString reason;
    QString authorizedBy;
    bool completed;
    
    QJsonObject toJson() const;
    static TransferRecord fromJson(const QJsonObject& json);
};

/**
 * @brief Patient manager - handles patient data, bed management, transfers
 */
class PatientManager : public QObject {
    Q_OBJECT

public:
    explicit PatientManager(QObject* parent = nullptr);
    ~PatientManager() override;

    // Patient management
    bool addPatient(const PatientInfo& patient);
    bool updatePatient(const QString& patientID, const PatientInfo& patient);
    bool removePatient(const QString& patientID);
    PatientInfo getPatient(const QString& patientID) const;
    QVector<PatientInfo> getAllPatients() const;
    QVector<PatientInfo> getActivePatients() const;
    QVector<PatientInfo> getPatientsByDepartment(const QString& department) const;
    QVector<PatientInfo> getPatientsByFloor(const QString& floor) const;
    QVector<PatientInfo> getPatientsByPriority(PatientInfo::Priority priority) const;
    bool patientExists(const QString& patientID) const;

    // Bed management
    bool addBed(const BedInfo& bed);
    bool updateBed(const QString& bedID, const BedInfo& bed);
    bool removeBed(const QString& bedID);
    BedInfo getBed(const QString& bedID) const;
    QVector<BedInfo> getAllBeds() const;
    QVector<BedInfo> getAvailableBeds(const QString& department = QString()) const;
    QVector<BedInfo> getOccupiedBeds(const QString& department = QString()) const;
    QVector<BedInfo> getBedsByDepartment(const QString& department) const;
    QVector<BedInfo> getBedsByFloor(const QString& floor) const;
    bool bedExists(const QString& bedID) const;
    bool isBedAvailable(const QString& bedID) const;

    // Bed assignment
    bool assignPatientToBed(const QString& patientID, const QString& bedID);
    bool unassignPatientFromBed(const QString& bedID);
    QString getPatientBed(const QString& patientID) const;
    QString getBedPatient(const QString& bedID) const;

    // Patient transfer
    QString initiateTransfer(const QString& patientID,
                           const QString& fromBed,
                           const QString& toBed,
                           const QString& reason,
                           const QString& authorizedBy);
    bool completeTransfer(const QString& transferID);
    bool cancelTransfer(const QString& transferID);
    QVector<TransferRecord> getPendingTransfers() const;
    QVector<TransferRecord> getTransferHistory(const QString& patientID) const;

    // Patient grouping
    QMap<QString, QVector<PatientInfo>> groupByDepartment() const;
    QMap<QString, QVector<PatientInfo>> groupByFloor() const;
    QMap<PatientInfo::Priority, QVector<PatientInfo>> groupByPriority() const;

    // Priority management
    void updatePatientPriority(const QString& patientID, PatientInfo::Priority priority);
    QVector<PatientInfo> getCriticalPatients() const;

    // Statistics
    int getTotalPatients() const;
    int getActivePatientCount() const;
    int getTotalBeds() const;
    int getAvailableBedCount() const;
    int getOccupiedBedCount() const;
    double getBedOccupancyRate() const;

    // Data persistence
    bool saveToFile(const QString& filename) const;
    bool loadFromFile(const QString& filename);

signals:
    void patientAdded(const QString& patientID);
    void patientUpdated(const QString& patientID);
    void patientRemoved(const QString& patientID);
    void patientPriorityChanged(const QString& patientID, PatientInfo::Priority priority);
    
    void bedAdded(const QString& bedID);
    void bedUpdated(const QString& bedID);
    void bedRemoved(const QString& bedID);
    void bedOccupancyChanged(const QString& bedID, bool occupied);
    
    void transferInitiated(const QString& transferID);
    void transferCompleted(const QString& transferID);
    void transferCancelled(const QString& transferID);

private:
    QString generateTransferID();
    void updateBedOccupancy();

    QMap<QString, PatientInfo> m_patients;
    QMap<QString, BedInfo> m_beds;
    QMap<QString, TransferRecord> m_transfers;
    
    // Quick lookup maps
    QMap<QString, QString> m_patientToBed;  // patientID -> bedID
    QMap<QString, QString> m_bedToPatient;  // bedID -> patientID
    
    int m_transferCounter;
};

} // namespace VitalStream

#endif // PATIENT_MANAGER_H
