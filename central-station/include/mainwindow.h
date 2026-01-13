#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QMap>
#include <memory>

class PatientGridView;
class AlarmManager;
class WebSocketClient;
class QLabel;
class QStatusBar;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onConnectionStatusChanged(bool connected);
    void onPatientDataReceived(const QString& patientId, const QByteArray& data);
    void onAlarmTriggered(const QString& patientId, int level, const QString& message);
    void onAlarmAcknowledged(const QString& alarmId);
    
    void showPatientDetails(const QString& patientId);
    void showTrendAnalysis(const QString& patientId);
    void show12LeadECG(const QString& patientId);
    void generateReport(const QString& patientId);
    
    void toggleDarkMode();
    void showSettings();
    void showAbout();

private:
    void setupUI();
    void setupMenuBar();
    void setupToolBar();
    void setupStatusBar();
    void setupConnections();
    void loadSettings();
    void saveSettings();
    
    void connectToBackend();
    void disconnectFromBackend();
    
    void updateConnectionStatus(bool connected);
    void updatePatientCount(int count);
    void updateAlarmCount(int count);

private:
    Ui::MainWindow *ui;
    
    std::unique_ptr<PatientGridView> m_patientGridView;
    std::unique_ptr<AlarmManager> m_alarmManager;
    std::unique_ptr<WebSocketClient> m_webSocketClient;
    
    QLabel *m_connectionStatusLabel;
    QLabel *m_patientCountLabel;
    QLabel *m_alarmCountLabel;
    
    QTimer *m_reconnectTimer;
    
    bool m_darkMode;
    QString m_backendUrl;
};

#endif // MAINWINDOW_H
