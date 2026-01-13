#include "quick_view_dashboard.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QStyleOption>
#include <algorithm>

namespace VitalStream {

// QuickViewPatientCard implementation

QuickViewPatientCard::QuickViewPatientCard(QWidget* parent)
    : QFrame(parent)
    , m_priority(Priority::Normal)
    , m_alarmActive(false)
{
    setupUI();
    setFrameStyle(QFrame::Box | QFrame::Raised);
    setLineWidth(2);
    setMinimumSize(200, 150);
    setCursor(Qt::PointingHandCursor);
}

void QuickViewPatientCard::setupUI() {
    auto* layout = new QVBoxLayout(this);
    layout->setSpacing(2);
    layout->setContentsMargins(5, 5, 5, 5);

    // Patient name and ID
    m_nameLabel = new QLabel("Patient Name");
    m_nameLabel->setStyleSheet("font-weight: bold; font-size: 12pt;");
    layout->addWidget(m_nameLabel);

    m_idLabel = new QLabel("ID: ---");
    m_idLabel->setStyleSheet("font-size: 9pt; color: gray;");
    layout->addWidget(m_idLabel);

    m_locationLabel = new QLabel("Room: --- Bed: ---");
    m_locationLabel->setStyleSheet("font-size: 9pt; color: gray;");
    layout->addWidget(m_locationLabel);

    layout->addSpacing(5);

    // Vital signs
    m_hrLabel = new QLabel("♥ HR: --- bpm");
    m_spo2Label = new QLabel("SpO₂: --- %");
    m_bpLabel = new QLabel("BP: ---/--- mmHg");
    m_rrLabel = new QLabel("RR: --- brpm");
    m_tempLabel = new QLabel("T: --- °C");

    layout->addWidget(m_hrLabel);
    layout->addWidget(m_spo2Label);
    layout->addWidget(m_bpLabel);
    layout->addWidget(m_rrLabel);
    layout->addWidget(m_tempLabel);

    layout->addStretch();

    // Alarm indicator
    m_alarmLabel = new QLabel("⚠ No Alarms");
    m_alarmLabel->setStyleSheet("font-size: 9pt;");
    layout->addWidget(m_alarmLabel);
}

void QuickViewPatientCard::setPatientName(const QString& name) {
    m_nameLabel->setText(name);
}

void QuickViewPatientCard::setPatientID(const QString& id) {
    m_patientID = id;
    m_idLabel->setText("ID: " + id);
}

void QuickViewPatientCard::setLocation(const QString& room, const QString& bed) {
    m_locationLabel->setText(QString("Room: %1 Bed: %2").arg(room, bed));
}

void QuickViewPatientCard::setPriority(Priority priority) {
    m_priority = priority;
    updatePriorityColor();
}

void QuickViewPatientCard::setHeartRate(int bpm) {
    m_hrLabel->setText(QString("♥ HR: %1 bpm").arg(bpm));
    if (bpm < 50 || bpm > 120) {
        m_hrLabel->setStyleSheet("color: red; font-weight: bold;");
    } else {
        m_hrLabel->setStyleSheet("");
    }
}

void QuickViewPatientCard::setSpO2(int percent) {
    m_spo2Label->setText(QString("SpO₂: %1 %").arg(percent));
    if (percent < 90) {
        m_spo2Label->setStyleSheet("color: red; font-weight: bold;");
    } else {
        m_spo2Label->setStyleSheet("");
    }
}

void QuickViewPatientCard::setBloodPressure(int systolic, int diastolic) {
    m_bpLabel->setText(QString("BP: %1/%2 mmHg").arg(systolic).arg(diastolic));
    if (systolic > 180 || systolic < 90 || diastolic > 110 || diastolic < 60) {
        m_bpLabel->setStyleSheet("color: red; font-weight: bold;");
    } else {
        m_bpLabel->setStyleSheet("");
    }
}

void QuickViewPatientCard::setRespiratoryRate(int brpm) {
    m_rrLabel->setText(QString("RR: %1 brpm").arg(brpm));
    if (brpm < 12 || brpm > 25) {
        m_rrLabel->setStyleSheet("color: red; font-weight: bold;");
    } else {
        m_rrLabel->setStyleSheet("");
    }
}

void QuickViewPatientCard::setTemperature(double celsius) {
    m_tempLabel->setText(QString("T: %1 °C").arg(celsius, 0, 'f', 1));
    if (celsius < 36.0 || celsius > 38.5) {
        m_tempLabel->setStyleSheet("color: red; font-weight: bold;");
    } else {
        m_tempLabel->setStyleSheet("");
    }
}

void QuickViewPatientCard::setAlarmCount(int count) {
    if (count > 0) {
        m_alarmLabel->setText(QString("⚠ %1 Active Alarm(s)").arg(count));
        m_alarmLabel->setStyleSheet("color: red; font-weight: bold;");
    } else {
        m_alarmLabel->setText("✓ No Alarms");
        m_alarmLabel->setStyleSheet("color: green;");
    }
}

void QuickViewPatientCard::setAlarmActive(bool active) {
    m_alarmActive = active;
    update();
}

void QuickViewPatientCard::updatePriorityColor() {
    QColor color = getPriorityColor();
    setStyleSheet(QString("QuickViewPatientCard { border: 3px solid %1; background-color: %2; }")
                  .arg(color.name())
                  .arg(color.lighter(180).name()));
}

QColor QuickViewPatientCard::getPriorityColor() const {
    switch (m_priority) {
    case Critical: return QColor(220, 20, 60);    // Crimson
    case High:     return QColor(255, 140, 0);    // Dark Orange
    case Medium:   return QColor(255, 215, 0);    // Gold
    case Low:      return QColor(50, 205, 50);    // Lime Green
    case Normal:   return QColor(200, 200, 200);  // Light Gray
    default:       return Qt::white;
    }
}

void QuickViewPatientCard::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        emit clicked();
    }
    QFrame::mousePressEvent(event);
}

void QuickViewPatientCard::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        emit doubleClicked();
    }
    QFrame::mouseDoubleClickEvent(event);
}

void QuickViewPatientCard::paintEvent(QPaintEvent* event) {
    QFrame::paintEvent(event);

    // Draw blinking alarm indicator if active
    if (m_alarmActive) {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        
        // Blink effect (simplified - would use timer in real implementation)
        static bool blink = false;
        if (blink) {
            painter.setBrush(QColor(255, 0, 0, 100));
            painter.setPen(Qt::NoPen);
            painter.drawRect(rect());
        }
        blink = !blink;
    }
}

// QuickViewDashboard implementation

QuickViewDashboard::QuickViewDashboard(QWidget* parent)
    : QWidget(parent)
    , m_gridSize(Grid_4x4)
    , m_rows(4)
    , m_cols(4)
    , m_autoRefresh(true)
    , m_refreshInterval(5)
{
    setupUI();

    // Setup auto-refresh timer
    m_refreshTimer = new QTimer(this);
    connect(m_refreshTimer, &QTimer::timeout, this, &QuickViewDashboard::onRefreshTimer);
    if (m_autoRefresh) {
        m_refreshTimer->start(m_refreshInterval * 1000);
    }
}

QuickViewDashboard::~QuickViewDashboard() {
}

void QuickViewDashboard::setupUI() {
    m_gridLayout = new QGridLayout(this);
    m_gridLayout->setSpacing(5);
    m_gridLayout->setContentsMargins(5, 5, 5, 5);
}

void QuickViewDashboard::setGridSize(GridSize size) {
    m_gridSize = size;

    switch (size) {
    case Grid_4x4:
        m_rows = 4; m_cols = 4;
        break;
    case Grid_4x6:
        m_rows = 4; m_cols = 6;
        break;
    case Grid_6x8:
        m_rows = 6; m_cols = 8;
        break;
    case Grid_8x8:
        m_rows = 8; m_cols = 8;
        break;
    }

    rebuildGrid();
}

void QuickViewDashboard::addPatient(const QString& patientID) {
    if (m_patientCards.contains(patientID)) {
        return;
    }

    auto* card = new QuickViewPatientCard(this);
    card->setPatientID(patientID);
    connect(card, &QuickViewPatientCard::clicked,
            this, &QuickViewDashboard::onPatientCardClicked);
    connect(card, &QuickViewPatientCard::doubleClicked,
            this, &QuickViewDashboard::onPatientCardDoubleClicked);

    m_patientCards[patientID] = card;
    m_patientOrder.append(patientID);

    updateCardPositions();
}

void QuickViewDashboard::removePatient(const QString& patientID) {
    if (!m_patientCards.contains(patientID)) {
        return;
    }

    auto* card = m_patientCards[patientID];
    m_gridLayout->removeWidget(card);
    delete card;

    m_patientCards.remove(patientID);
    m_patientOrder.removeAll(patientID);

    updateCardPositions();
}

void QuickViewDashboard::updatePatient(const QString& patientID) {
    // Trigger data refresh for specific patient
    emit refreshRequested();
}

void QuickViewDashboard::clearAllPatients() {
    for (auto* card : m_patientCards) {
        m_gridLayout->removeWidget(card);
        delete card;
    }
    m_patientCards.clear();
    m_patientOrder.clear();
}

void QuickViewDashboard::updatePatientInfo(const QString& patientID,
                                          const QString& name,
                                          const QString& room,
                                          const QString& bed) {
    auto* card = getPatientCard(patientID);
    if (card) {
        card->setPatientName(name);
        card->setLocation(room, bed);
    }
}

void QuickViewDashboard::updateVitalSigns(const QString& patientID,
                                         int hr, int spo2, int sysBP, int diaBP,
                                         int rr, double temp) {
    auto* card = getPatientCard(patientID);
    if (card) {
        card->setHeartRate(hr);
        card->setSpO2(spo2);
        card->setBloodPressure(sysBP, diaBP);
        card->setRespiratoryRate(rr);
        card->setTemperature(temp);
    }
}

void QuickViewDashboard::updatePriority(const QString& patientID,
                                       QuickViewPatientCard::Priority priority) {
    auto* card = getPatientCard(patientID);
    if (card) {
        card->setPriority(priority);
    }
}

void QuickViewDashboard::updateAlarmStatus(const QString& patientID,
                                          int alarmCount, bool active) {
    auto* card = getPatientCard(patientID);
    if (card) {
        card->setAlarmCount(alarmCount);
        card->setAlarmActive(active);
    }
}

void QuickViewDashboard::sortByPriority() {
    std::sort(m_patientOrder.begin(), m_patientOrder.end(),
              [this](const QString& a, const QString& b) {
                  auto* cardA = m_patientCards[a];
                  auto* cardB = m_patientCards[b];
                  return cardA->priority() < cardB->priority();
              });
    updateCardPositions();
}

void QuickViewDashboard::sortByLocation() {
    // Would sort by room/bed
    updateCardPositions();
}

void QuickViewDashboard::sortByName() {
    // Would sort by patient name
    updateCardPositions();
}

void QuickViewDashboard::filterByDepartment(const QString& department) {
    m_currentFilter = department;
    // Would filter cards
    updateCardPositions();
}

void QuickViewDashboard::clearFilter() {
    m_currentFilter.clear();
    updateCardPositions();
}

void QuickViewDashboard::setAutoRefresh(bool enable) {
    m_autoRefresh = enable;
    if (enable) {
        m_refreshTimer->start(m_refreshInterval * 1000);
    } else {
        m_refreshTimer->stop();
    }
}

void QuickViewDashboard::setRefreshInterval(int seconds) {
    m_refreshInterval = seconds;
    if (m_autoRefresh) {
        m_refreshTimer->start(m_refreshInterval * 1000);
    }
}

void QuickViewDashboard::rebuildGrid() {
    updateCardPositions();
}

void QuickViewDashboard::updateCardPositions() {
    // Remove all cards from layout
    for (auto* card : m_patientCards) {
        m_gridLayout->removeWidget(card);
    }

    // Re-add cards in order
    int index = 0;
    for (const QString& patientID : m_patientOrder) {
        if (index >= m_rows * m_cols) {
            break; // Grid full
        }

        auto* card = m_patientCards[patientID];
        int row = index / m_cols;
        int col = index % m_cols;
        m_gridLayout->addWidget(card, row, col);
        card->show();
        index++;
    }
}

QuickViewPatientCard* QuickViewDashboard::getPatientCard(const QString& patientID) {
    return m_patientCards.value(patientID, nullptr);
}

void QuickViewDashboard::onPatientCardClicked() {
    auto* card = qobject_cast<QuickViewPatientCard*>(sender());
    if (card) {
        emit patientSelected(card->patientID());
    }
}

void QuickViewDashboard::onPatientCardDoubleClicked() {
    auto* card = qobject_cast<QuickViewPatientCard*>(sender());
    if (card) {
        emit patientDoubleClicked(card->patientID());
    }
}

void QuickViewDashboard::onRefreshTimer() {
    emit refreshRequested();
}

} // namespace VitalStream
