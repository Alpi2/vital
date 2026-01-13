#include "ecg_12lead_widget.h"
#include <QGridLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QPrinter>
#include <QPrintDialog>
#include <QPainter>
#include <QFileDialog>
#include <QMessageBox>

namespace VitalStream {

ECG12LeadWidget::ECG12LeadWidget(QWidget* parent)
    : QWidget(parent)
    , m_displayFormat(DisplayFormat::Standard_3x4)
{
    setupUI();
    setupLeadRenderers();
    arrangeLeads();
}

ECG12LeadWidget::~ECG12LeadWidget() {
    // Renderers will be deleted by Qt parent-child relationship
}

void ECG12LeadWidget::setupUI() {
    m_mainLayout = new QVBoxLayout(this);
    m_mainLayout->setSpacing(5);
    m_mainLayout->setContentsMargins(5, 5, 5, 5);

    // Header with controls
    m_headerWidget = new QWidget(this);
    auto* headerLayout = new QHBoxLayout(m_headerWidget);

    // Format selector
    headerLayout->addWidget(new QLabel("Format:"));
    m_formatComboBox = new QComboBox();
    m_formatComboBox->addItem("Standard 3x4", Standard_3x4);
    m_formatComboBox->addItem("Cabrera", Cabrera);
    m_formatComboBox->addItem("6x2 Format", Format_6x2);
    m_formatComboBox->addItem("Rhythm Strip", Rhythm_Strip);
    connect(m_formatComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ECG12LeadWidget::onFormatChanged);
    headerLayout->addWidget(m_formatComboBox);

    headerLayout->addStretch();

    // Print button
    m_printButton = new QPushButton("Print");
    connect(m_printButton, &QPushButton::clicked, this, &ECG12LeadWidget::onPrintClicked);
    headerLayout->addWidget(m_printButton);

    // Export button
    m_exportButton = new QPushButton("Export PDF");
    connect(m_exportButton, &QPushButton::clicked, this, &ECG12LeadWidget::onExportClicked);
    headerLayout->addWidget(m_exportButton);

    m_mainLayout->addWidget(m_headerWidget);

    // Waveform container
    m_waveformContainer = new QWidget(this);
    m_mainLayout->addWidget(m_waveformContainer, 1);

    // Measurement panel
    m_measurementWidget = new QWidget(this);
    auto* measurementLayout = new QHBoxLayout(m_measurementWidget);
    measurementLayout->setSpacing(15);

    m_heartRateLabel = new QLabel("HR: -- bpm");
    m_prIntervalLabel = new QLabel("PR: -- ms");
    m_qrsDurationLabel = new QLabel("QRS: -- ms");
    m_qtIntervalLabel = new QLabel("QT: -- ms");
    m_qtcIntervalLabel = new QLabel("QTc: -- ms");
    m_axisLabel = new QLabel("Axis: -- °");

    measurementLayout->addWidget(m_heartRateLabel);
    measurementLayout->addWidget(m_prIntervalLabel);
    measurementLayout->addWidget(m_qrsDurationLabel);
    measurementLayout->addWidget(m_qtIntervalLabel);
    measurementLayout->addWidget(m_qtcIntervalLabel);
    measurementLayout->addWidget(m_axisLabel);
    measurementLayout->addStretch();

    m_mainLayout->addWidget(m_measurementWidget);

    // Interpretation panel
    m_interpretationWidget = new QWidget(this);
    auto* interpretationLayout = new QVBoxLayout(m_interpretationWidget);
    m_interpretationLabel = new QLabel("Interpretation: Normal sinus rhythm");
    m_interpretationLabel->setWordWrap(true);
    interpretationLayout->addWidget(m_interpretationLabel);

    m_mainLayout->addWidget(m_interpretationWidget);
}

void ECG12LeadWidget::setupLeadRenderers() {
    // Create renderers for all 12 leads
    QStringList leads = {"I", "II", "III", "aVR", "aVL", "aVF",
                         "V1", "V2", "V3", "V4", "V5", "V6"};

    QList<QColor> colors = {
        Qt::green,    // I
        Qt::green,    // II
        Qt::green,    // III
        Qt::cyan,     // aVR
        Qt::cyan,     // aVL
        Qt::cyan,     // aVF
        Qt::yellow,   // V1
        Qt::yellow,   // V2
        Qt::yellow,   // V3
        Qt::yellow,   // V4
        Qt::yellow,   // V5
        Qt::yellow    // V6
    };

    for (int i = 0; i < leads.size(); ++i) {
        auto* renderer = new WaveformRenderer(this);
        renderer->addChannel(leads[i], colors[i]);
        renderer->setTimeWindow(2.5f);  // 2.5 seconds for 12-lead
        renderer->setSampleRate(500.0f); // 500 Hz for ECG
        renderer->setGridVisible(true);
        renderer->setMinimumHeight(80);

        m_leadRenderers[leads[i]] = renderer;
    }
}

void ECG12LeadWidget::arrangeLeads() {
    // Clear existing layout
    if (m_waveformContainer->layout()) {
        QLayoutItem* item;
        while ((item = m_waveformContainer->layout()->takeAt(0)) != nullptr) {
            // Don't delete widgets, just remove from layout
        }
        delete m_waveformContainer->layout();
    }

    QGridLayout* gridLayout = new QGridLayout(m_waveformContainer);
    gridLayout->setSpacing(2);
    gridLayout->setContentsMargins(0, 0, 0, 0);

    QStringList leadOrder;

    switch (m_displayFormat) {
    case Standard_3x4:
        leadOrder = standardLeadOrder();
        // 3 columns, 4 rows + rhythm strip
        for (int i = 0; i < 12; ++i) {
            int row = i / 3;
            int col = i % 3;
            gridLayout->addWidget(m_leadRenderers[leadOrder[i]], row, col);
        }
        // Add rhythm strip (Lead II) at bottom
        gridLayout->addWidget(m_leadRenderers["II"], 4, 0, 1, 3);
        break;

    case Cabrera:
        leadOrder = cabreraLeadOrder();
        // Cabrera format: anatomical ordering
        // Row 0: aVL, I, -aVR
        gridLayout->addWidget(m_leadRenderers["aVL"], 0, 0);
        gridLayout->addWidget(m_leadRenderers["I"], 0, 1);
        gridLayout->addWidget(m_leadRenderers["aVR"], 0, 2); // Note: should be inverted
        // Row 1: II, aVF, III
        gridLayout->addWidget(m_leadRenderers["II"], 1, 0);
        gridLayout->addWidget(m_leadRenderers["aVF"], 1, 1);
        gridLayout->addWidget(m_leadRenderers["III"], 1, 2);
        // Row 2-3: V1-V6
        for (int i = 0; i < 6; ++i) {
            int row = 2 + i / 3;
            int col = i % 3;
            gridLayout->addWidget(m_leadRenderers[QString("V%1").arg(i + 1)], row, col);
        }
        break;

    case Format_6x2:
        // 6 columns, 2 rows
        for (int i = 0; i < 12; ++i) {
            int row = i / 6;
            int col = i % 6;
            gridLayout->addWidget(m_leadRenderers[standardLeadOrder()[i]], row, col);
        }
        break;

    case Rhythm_Strip:
        // Single long rhythm strip (Lead II)
        gridLayout->addWidget(m_leadRenderers["II"], 0, 0);
        m_leadRenderers["II"]->setTimeWindow(10.0f); // 10 seconds
        break;

    default:
        break;
    }
}

QStringList ECG12LeadWidget::standardLeadOrder() {
    return {"I", "aVR", "V1", "V4",
            "II", "aVL", "V2", "V5",
            "III", "aVF", "V3", "V6"};
}

QStringList ECG12LeadWidget::cabreraLeadOrder() {
    return {"aVL", "I", "aVR", "II", "aVF", "III",
            "V1", "V2", "V3", "V4", "V5", "V6"};
}

void ECG12LeadWidget::updateECGData(const QString& leadName, const QVector<float>& data) {
    if (m_leadRenderers.contains(leadName)) {
        m_leadRenderers[leadName]->addDataPoints(0, data);
    }
}

void ECG12LeadWidget::updateAllLeads(const QMap<QString, QVector<float>>& allLeadsData) {
    for (auto it = allLeadsData.begin(); it != allLeadsData.end(); ++it) {
        updateECGData(it.key(), it.value());
    }
}

void ECG12LeadWidget::setDisplayFormat(DisplayFormat format) {
    if (m_displayFormat != format) {
        m_displayFormat = format;
        arrangeLeads();
        emit formatChanged(format);
    }
}

void ECG12LeadWidget::setHeartRate(int bpm) {
    m_heartRateLabel->setText(QString("HR: %1 bpm").arg(bpm));
}

void ECG12LeadWidget::setPRInterval(int ms) {
    m_prIntervalLabel->setText(QString("PR: %1 ms").arg(ms));
}

void ECG12LeadWidget::setQRSDuration(int ms) {
    m_qrsDurationLabel->setText(QString("QRS: %1 ms").arg(ms));
}

void ECG12LeadWidget::setQTInterval(int ms) {
    m_qtIntervalLabel->setText(QString("QT: %1 ms").arg(ms));
}

void ECG12LeadWidget::setQTcInterval(int ms) {
    m_qtcIntervalLabel->setText(QString("QTc: %1 ms").arg(ms));
}

void ECG12LeadWidget::setAxis(int degrees) {
    m_axisLabel->setText(QString("Axis: %1°").arg(degrees));
}

void ECG12LeadWidget::setInterpretation(const QString& text) {
    m_interpretationLabel->setText("Interpretation: " + text);
}

void ECG12LeadWidget::addFinding(const QString& finding) {
    m_findings.append(finding);
    QString allFindings = m_findings.join(", ");
    m_interpretationLabel->setText("Findings: " + allFindings);
}

void ECG12LeadWidget::clearFindings() {
    m_findings.clear();
    m_interpretationLabel->setText("Interpretation: Normal sinus rhythm");
}

void ECG12LeadWidget::printECG() {
    QPrinter printer(QPrinter::HighResolution);
    QPrintDialog dialog(&printer, this);

    if (dialog.exec() == QDialog::Accepted) {
        QPainter painter(&printer);
        render(&painter);
    }
}

QImage ECG12LeadWidget::captureECG() {
    QImage image(size(), QImage::Format_ARGB32);
    render(&image);
    return image;
}

void ECG12LeadWidget::exportToPDF(const QString& filename) {
    QString file = filename;
    if (file.isEmpty()) {
        file = QFileDialog::getSaveFileName(this, "Export ECG to PDF",
                                            QString(), "PDF Files (*.pdf)");
    }

    if (!file.isEmpty()) {
        QPrinter printer(QPrinter::HighResolution);
        printer.setOutputFormat(QPrinter::PdfFormat);
        printer.setOutputFileName(file);
        printer.setPageSize(QPageSize::A4);

        QPainter painter(&printer);
        render(&painter);

        QMessageBox::information(this, "Export Complete",
                                 "ECG exported to PDF successfully.");
    }
}

void ECG12LeadWidget::onFormatChanged(int index) {
    DisplayFormat format = static_cast<DisplayFormat>(
        m_formatComboBox->itemData(index).toInt());
    setDisplayFormat(format);
}

void ECG12LeadWidget::onPrintClicked() {
    printECG();
}

void ECG12LeadWidget::onExportClicked() {
    exportToPDF();
}

} // namespace VitalStream
