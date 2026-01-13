#ifndef QUICK_VIEW_DASHBOARD_H
#define QUICK_VIEW_DASHBOARD_H

#include <QWidget>
#include <QGridLayout>
#include <QLabel>
#include <QFrame>
#include <QPushButton>
#include <QTimer>
#include <QVector>
#include <QMap>

namespace VitalStream {

/**
 * @brief Quick view patient card
 */
class QuickViewPatientCard : public QFrame {
    Q_OBJECT

public:
    enum Priority {
        Critical,   // Red
        High,       // Orange
        Medium,     // Yellow
        Low,        // Green
        Normal      // White
    };

    explicit QuickViewPatientCard(QWidget* parent = nullptr);

    // Patient info
    void setPatientName(const QString& name);
    void setPatientID(const QString& id);
    void setLocation(const QString& room, const QString& bed);
    void setPriority(Priority priority);

    // Vital signs (quick view)
    void setHeartRate(int bpm);
    void setSpO2(int percent);
    void setBloodPressure(int systolic, int diastolic);
    void setRespiratoryRate(int brpm);
    void setTemperature(double celsius);

    // Alarm status
    void setAlarmCount(int count);
    void setAlarmActive(bool active);

    QString patientID() const { return m_patientID; }
    Priority priority() const { return m_priority; }

signals:
    void clicked();
    void doubleClicked();

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void paintEvent(QPaintEvent* event) override;

private:
    void setupUI();
    void updatePriorityColor();
    QColor getPriorityColor() const;

    QString m_patientID;
    Priority m_priority;
    bool m_alarmActive;

    // UI components
    QLabel* m_nameLabel;
    QLabel* m_idLabel;
    QLabel* m_locationLabel;
    QLabel* m_hrLabel;
    QLabel* m_spo2Label;
    QLabel* m_bpLabel;
    QLabel* m_rrLabel;
    QLabel* m_tempLabel;
    QLabel* m_alarmLabel;
};

/**
 * @brief Quick view dashboard - overview of all patients
 * 
 * Features:
 * - Grid layout (4x4, 4x6, 6x8, etc.)
 * - Color-coded by priority
 * - Real-time vital signs
 * - Alarm indicators
 * - Click to view details
 * - Auto-refresh
 * - Sorting by priority
 */
class QuickViewDashboard : public QWidget {
    Q_OBJECT

public:
    enum GridSize {
        Grid_4x4,   // 16 patients
        Grid_4x6,   // 24 patients
        Grid_6x8,   // 48 patients
        Grid_8x8    // 64 patients
    };

    explicit QuickViewDashboard(QWidget* parent = nullptr);
    ~QuickViewDashboard() override;

    // Grid configuration
    void setGridSize(GridSize size);
    GridSize gridSize() const { return m_gridSize; }

    // Patient management
    void addPatient(const QString& patientID);
    void removePatient(const QString& patientID);
    void updatePatient(const QString& patientID);
    void clearAllPatients();

    // Update patient data
    void updatePatientInfo(const QString& patientID,
                          const QString& name,
                          const QString& room,
                          const QString& bed);
    void updateVitalSigns(const QString& patientID,
                         int hr, int spo2, int sysBP, int diaBP,
                         int rr, double temp);
    void updatePriority(const QString& patientID,
                       QuickViewPatientCard::Priority priority);
    void updateAlarmStatus(const QString& patientID, int alarmCount, bool active);

    // Sorting and filtering
    void sortByPriority();
    void sortByLocation();
    void sortByName();
    void filterByDepartment(const QString& department);
    void clearFilter();

    // Auto-refresh
    void setAutoRefresh(bool enable);
    void setRefreshInterval(int seconds);

signals:
    void patientSelected(const QString& patientID);
    void patientDoubleClicked(const QString& patientID);
    void refreshRequested();

private slots:
    void onPatientCardClicked();
    void onPatientCardDoubleClicked();
    void onRefreshTimer();

private:
    void setupUI();
    void rebuildGrid();
    void updateCardPositions();
    QuickViewPatientCard* getPatientCard(const QString& patientID);

    GridSize m_gridSize;
    int m_rows;
    int m_cols;

    QGridLayout* m_gridLayout;
    QMap<QString, QuickViewPatientCard*> m_patientCards;
    QVector<QString> m_patientOrder;

    // Auto-refresh
    QTimer* m_refreshTimer;
    bool m_autoRefresh;
    int m_refreshInterval;

    // Filtering
    QString m_currentFilter;
};

} // namespace VitalStream

#endif // QUICK_VIEW_DASHBOARD_H
