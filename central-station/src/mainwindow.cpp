#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "patientgridview.h"
#include "alarmmanager.h"
#include "websocketclient.h"

#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QLabel>
#include <QSettings>
#include <QMessageBox>
#include <QPalette>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , m_darkMode(false)
    , m_backendUrl("ws://localhost:8080")
{
    ui->setupUi(this);
    
    setupUI();
    setupMenuBar();
    setupToolBar();
    setupStatusBar();
    
    loadSettings();
    setupConnections();
    
    // Connect to backend
    connectToBackend();
}

MainWindow::~MainWindow()
{
    saveSettings();
    delete ui;
}

void MainWindow::setupUI()
{
    setWindowTitle("VitalStream Central Station");
    setMinimumSize(1920, 1080);
    
    // Create main components
    m_patientGridView = std::make_unique<PatientGridView>(this);
    m_alarmManager = std::make_unique<AlarmManager>(this);
    m_webSocketClient = std::make_unique<WebSocketClient>(this);
    
    // Set central widget
    setCentralWidget(m_patientGridView.get());
    
    // Setup reconnect timer
    m_reconnectTimer = new QTimer(this);
    m_reconnectTimer->setInterval(5000); // 5 seconds
}

void MainWindow::setupMenuBar()
{
    // File menu
    QMenu *fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(tr("&Settings"), this, &MainWindow::showSettings, QKeySequence::Preferences);
    fileMenu->addSeparator();
    fileMenu->addAction(tr("E&xit"), this, &QWidget::close, QKeySequence::Quit);
    
    // View menu
    QMenu *viewMenu = menuBar()->addMenu(tr("&View"));
    viewMenu->addAction(tr("&Dark Mode"), this, &MainWindow::toggleDarkMode, QKeySequence("Ctrl+D"));
    viewMenu->addSeparator();
    viewMenu->addAction(tr("&Full Screen"), this, [this]() {
        if (isFullScreen()) {
            showNormal();
        } else {
            showFullScreen();
        }
    }, QKeySequence::FullScreen);
    
    // Patient menu
    QMenu *patientMenu = menuBar()->addMenu(tr("&Patient"));
    patientMenu->addAction(tr("&Details"), this, [this]() {
        // Show details for selected patient
    }, QKeySequence("Ctrl+I"));
    patientMenu->addAction(tr("&Trend Analysis"), this, [this]() {
        // Show trend analysis
    }, QKeySequence("Ctrl+T"));
    patientMenu->addAction(tr("&12-Lead ECG"), this, [this]() {
        // Show 12-lead ECG
    }, QKeySequence("Ctrl+E"));
    
    // Help menu
    QMenu *helpMenu = menuBar()->addMenu(tr("&Help"));
    helpMenu->addAction(tr("&About"), this, &MainWindow::showAbout);
}

void MainWindow::setupToolBar()
{
    QToolBar *toolbar = addToolBar(tr("Main Toolbar"));
    toolbar->setMovable(false);
    
    // Add quick access buttons
    toolbar->addAction(tr("Connect"), this, &MainWindow::connectToBackend);
    toolbar->addAction(tr("Disconnect"), this, &MainWindow::disconnectFromBackend);
    toolbar->addSeparator();
    toolbar->addAction(tr("Acknowledge All Alarms"), this, [this]() {
        m_alarmManager->acknowledgeAllAlarms();
    });
}

void MainWindow::setupStatusBar()
{
    m_connectionStatusLabel = new QLabel(tr("Disconnected"));
    m_patientCountLabel = new QLabel(tr("Patients: 0"));
    m_alarmCountLabel = new QLabel(tr("Alarms: 0"));
    
    statusBar()->addPermanentWidget(m_connectionStatusLabel);
    statusBar()->addPermanentWidget(m_patientCountLabel);
    statusBar()->addPermanentWidget(m_alarmCountLabel);
}

void MainWindow::setupConnections()
{
    // WebSocket connections
    connect(m_webSocketClient.get(), &WebSocketClient::connected,
            this, [this]() { onConnectionStatusChanged(true); });
    connect(m_webSocketClient.get(), &WebSocketClient::disconnected,
            this, [this]() { onConnectionStatusChanged(false); });
    connect(m_webSocketClient.get(), &WebSocketClient::dataReceived,
            this, &MainWindow::onPatientDataReceived);
    
    // Alarm connections
    connect(m_alarmManager.get(), &AlarmManager::alarmTriggered,
            this, &MainWindow::onAlarmTriggered);
    connect(m_alarmManager.get(), &AlarmManager::alarmAcknowledged,
            this, &MainWindow::onAlarmAcknowledged);
    
    // Reconnect timer
    connect(m_reconnectTimer, &QTimer::timeout,
            this, &MainWindow::connectToBackend);
}

void MainWindow::loadSettings()
{
    QSettings settings;
    m_backendUrl = settings.value("backend/url", "ws://localhost:8080").toString();
    m_darkMode = settings.value("ui/darkMode", false).toBool();
    
    if (m_darkMode) {
        toggleDarkMode();
    }
    
    restoreGeometry(settings.value("window/geometry").toByteArray());
    restoreState(settings.value("window/state").toByteArray());
}

void MainWindow::saveSettings()
{
    QSettings settings;
    settings.setValue("backend/url", m_backendUrl);
    settings.setValue("ui/darkMode", m_darkMode);
    settings.setValue("window/geometry", saveGeometry());
    settings.setValue("window/state", saveState());
}

void MainWindow::connectToBackend()
{
    m_webSocketClient->connectToServer(m_backendUrl);
    m_reconnectTimer->stop();
}

void MainWindow::disconnectFromBackend()
{
    m_webSocketClient->disconnectFromServer();
    m_reconnectTimer->stop();
}

void MainWindow::onConnectionStatusChanged(bool connected)
{
    updateConnectionStatus(connected);
    
    if (!connected) {
        // Start reconnect timer
        m_reconnectTimer->start();
    }
}

void MainWindow::onPatientDataReceived(const QString& patientId, const QByteArray& data)
{
    // Forward to patient grid view
    m_patientGridView->updatePatientData(patientId, data);
    
    // Update patient count
    updatePatientCount(m_patientGridView->getPatientCount());
}

void MainWindow::onAlarmTriggered(const QString& patientId, int level, const QString& message)
{
    // Update alarm count
    updateAlarmCount(m_alarmManager->getActiveAlarmCount());
    
    // Show notification
    statusBar()->showMessage(tr("Alarm: %1 - %2").arg(patientId, message), 5000);
}

void MainWindow::onAlarmAcknowledged(const QString& alarmId)
{
    updateAlarmCount(m_alarmManager->getActiveAlarmCount());
}

void MainWindow::updateConnectionStatus(bool connected)
{
    if (connected) {
        m_connectionStatusLabel->setText(tr("✅ Connected"));
        m_connectionStatusLabel->setStyleSheet("color: green;");
    } else {
        m_connectionStatusLabel->setText(tr("❌ Disconnected"));
        m_connectionStatusLabel->setStyleSheet("color: red;");
    }
}

void MainWindow::updatePatientCount(int count)
{
    m_patientCountLabel->setText(tr("Patients: %1").arg(count));
}

void MainWindow::updateAlarmCount(int count)
{
    m_alarmCountLabel->setText(tr("Alarms: %1").arg(count));
    
    if (count > 0) {
        m_alarmCountLabel->setStyleSheet("color: red; font-weight: bold;");
    } else {
        m_alarmCountLabel->setStyleSheet("");
    }
}

void MainWindow::toggleDarkMode()
{
    m_darkMode = !m_darkMode;
    
    QPalette palette;
    if (m_darkMode) {
        // Dark mode colors
        palette.setColor(QPalette::Window, QColor(53, 53, 53));
        palette.setColor(QPalette::WindowText, Qt::white);
        palette.setColor(QPalette::Base, QColor(25, 25, 25));
        palette.setColor(QPalette::AlternateBase, QColor(53, 53, 53));
        palette.setColor(QPalette::ToolTipBase, Qt::white);
        palette.setColor(QPalette::ToolTipText, Qt::white);
        palette.setColor(QPalette::Text, Qt::white);
        palette.setColor(QPalette::Button, QColor(53, 53, 53));
        palette.setColor(QPalette::ButtonText, Qt::white);
        palette.setColor(QPalette::BrightText, Qt::red);
        palette.setColor(QPalette::Link, QColor(42, 130, 218));
        palette.setColor(QPalette::Highlight, QColor(42, 130, 218));
        palette.setColor(QPalette::HighlightedText, Qt::black);
    } else {
        // Light mode (default)
        palette = qApp->style()->standardPalette();
    }
    
    qApp->setPalette(palette);
}

void MainWindow::showSettings()
{
    // TODO: Implement settings dialog
    QMessageBox::information(this, tr("Settings"), tr("Settings dialog not yet implemented."));
}

void MainWindow::showAbout()
{
    QMessageBox::about(this, tr("About VitalStream Central Station"),
        tr("<h2>VitalStream Central Station v1.0.0</h2>"
           "<p>Professional multi-patient monitoring system</p>"
           "<p>Copyright © 2026 VitalStream Medical Systems</p>"
           "<p>Built with Qt %1</p>").arg(QT_VERSION_STR));
}

void MainWindow::showPatientDetails(const QString& patientId)
{
    // TODO: Implement patient details dialog
}

void MainWindow::showTrendAnalysis(const QString& patientId)
{
    // TODO: Implement trend analysis window
}

void MainWindow::show12LeadECG(const QString& patientId)
{
    // TODO: Implement 12-lead ECG window
}

void MainWindow::generateReport(const QString& patientId)
{
    // TODO: Implement PDF report generation
}
