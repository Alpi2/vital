#include "trend_chart_widget.h"
#include <QHBoxLayout>
#include <QLabel>
#include <QFileDialog>
#include <QTextStream>
#include <QPrinter>
#include <QPainter>
#include <QMessageBox>
#include <QHeaderView>
#include <QScatterSeries>
#include <QBarSeries>
#include <QBarSet>
#include <QBarCategoryAxis>
#include <algorithm>

namespace VitalStream {

TrendChartWidget::TrendChartWidget(QWidget* parent)
    : QWidget(parent)
    , m_viewMode(ViewMode::LineChart)
    , m_timeWindow(TimeWindow::TwentyFourHours)
    , m_chart(nullptr)
    , m_chartView(nullptr)
    , m_axisX(nullptr)
    , m_tableWidget(nullptr)
{
    setupUI();
    setupChart();
    setupTable();

    // Initialize time range
    m_endTime = QDateTime::currentDateTime();
    m_startTime = m_endTime.addSecs(-m_timeWindow * 3600);

    // Add some default vital signs
    addVitalSign("Heart Rate", Qt::red, 40, 180, "bpm");
    addVitalSign("SpO2", Qt::blue, 80, 100, "%");
    addVitalSign("Respiratory Rate", Qt::green, 8, 40, "brpm");
    addVitalSign("Blood Pressure (Sys)", Qt::magenta, 60, 200, "mmHg");
    addVitalSign("Temperature", Qt::yellow, 35, 42, "°C");
}

TrendChartWidget::~TrendChartWidget() {
}

void TrendChartWidget::setupUI() {
    m_mainLayout = new QVBoxLayout(this);
    m_mainLayout->setSpacing(5);
    m_mainLayout->setContentsMargins(5, 5, 5, 5);

    // Control panel
    m_controlPanel = new QWidget(this);
    auto* controlLayout = new QHBoxLayout(m_controlPanel);

    // Time window selector
    controlLayout->addWidget(new QLabel("Time Window:"));
    m_timeWindowCombo = new QComboBox();
    m_timeWindowCombo->addItem("1 Hour", OneHour);
    m_timeWindowCombo->addItem("6 Hours", SixHours);
    m_timeWindowCombo->addItem("12 Hours", TwelveHours);
    m_timeWindowCombo->addItem("24 Hours", TwentyFourHours);
    m_timeWindowCombo->addItem("48 Hours", FortyEightHours);
    m_timeWindowCombo->addItem("72 Hours", SeventyTwoHours);
    m_timeWindowCombo->setCurrentIndex(3); // 24 hours default
    connect(m_timeWindowCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &TrendChartWidget::onTimeWindowChanged);
    controlLayout->addWidget(m_timeWindowCombo);

    controlLayout->addSpacing(20);

    // View mode selector
    controlLayout->addWidget(new QLabel("View:"));
    m_viewModeCombo = new QComboBox();
    m_viewModeCombo->addItem("Line Chart", LineChart);
    m_viewModeCombo->addItem("Table", TabularView);
    m_viewModeCombo->addItem("Histogram", Histogram);
    m_viewModeCombo->addItem("Scatter Plot", ScatterPlot);
    connect(m_viewModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &TrendChartWidget::onViewModeChanged);
    controlLayout->addWidget(m_viewModeCombo);

    controlLayout->addStretch();

    // Export buttons
    m_exportCSVButton = new QPushButton("Export CSV");
    connect(m_exportCSVButton, &QPushButton::clicked,
            this, &TrendChartWidget::onExportCSVClicked);
    controlLayout->addWidget(m_exportCSVButton);

    m_exportPDFButton = new QPushButton("Export PDF");
    connect(m_exportPDFButton, &QPushButton::clicked,
            this, &TrendChartWidget::onExportPDFClicked);
    controlLayout->addWidget(m_exportPDFButton);

    m_mainLayout->addWidget(m_controlPanel);
}

void TrendChartWidget::setupChart() {
    m_chart = new QChart();
    m_chart->setTitle("Vital Signs Trends");
    m_chart->setAnimationOptions(QChart::SeriesAnimations);
    m_chart->legend()->setVisible(true);
    m_chart->legend()->setAlignment(Qt::AlignBottom);

    // X-axis (time)
    m_axisX = new QDateTimeAxis();
    m_axisX->setFormat("hh:mm");
    m_axisX->setTitleText("Time");
    m_chart->addAxis(m_axisX, Qt::AlignBottom);

    m_chartView = new QChartView(m_chart);
    m_chartView->setRenderHint(QPainter::Antialiasing);
    m_chartView->setRubberBand(QChartView::RectangleRubberBand);

    m_mainLayout->addWidget(m_chartView, 1);
}

void TrendChartWidget::setupTable() {
    m_tableWidget = new QTableWidget(this);
    m_tableWidget->setColumnCount(3);
    m_tableWidget->setHorizontalHeaderLabels({"Time", "Vital Sign", "Value"});
    m_tableWidget->horizontalHeader()->setStretchLastSection(true);
    m_tableWidget->setAlternatingRowColors(true);
    m_tableWidget->setSortingEnabled(true);
    m_tableWidget->hide(); // Hidden by default

    m_mainLayout->addWidget(m_tableWidget, 1);
}

void TrendChartWidget::addVitalSign(const QString& name, const QColor& color,
                                   double minValue, double maxValue, const QString& unit) {
    VitalSignConfig config;
    config.name = name;
    config.color = color;
    config.minValue = minValue;
    config.maxValue = maxValue;
    config.unit = unit;
    config.visible = true;

    // Create series
    config.series = new QLineSeries();
    config.series->setName(name);
    config.series->setColor(color);

    m_chart->addSeries(config.series);
    config.series->attachAxis(m_axisX);

    // Create Y-axis
    auto* axisY = new QValueAxis();
    axisY->setTitleText(name + " (" + unit + ")");
    axisY->setRange(minValue, maxValue);
    axisY->setLabelFormat("%.0f");
    m_chart->addAxis(axisY, Qt::AlignLeft);
    config.series->attachAxis(axisY);

    m_axisYMap[name] = axisY;
    m_vitalSigns[name] = config;
}

void TrendChartWidget::removeVitalSign(const QString& name) {
    if (m_vitalSigns.contains(name)) {
        auto& config = m_vitalSigns[name];
        m_chart->removeSeries(config.series);
        delete config.series;

        if (m_axisYMap.contains(name)) {
            m_chart->removeAxis(m_axisYMap[name]);
            delete m_axisYMap[name];
            m_axisYMap.remove(name);
        }

        m_vitalSigns.remove(name);
    }
}

void TrendChartWidget::setVitalSignVisible(const QString& name, bool visible) {
    if (m_vitalSigns.contains(name)) {
        m_vitalSigns[name].visible = visible;
        m_vitalSigns[name].series->setVisible(visible);
        if (m_axisYMap.contains(name)) {
            m_axisYMap[name]->setVisible(visible);
        }
    }
}

void TrendChartWidget::addDataPoint(const QString& vitalSign,
                                   const QDateTime& timestamp, double value) {
    if (!m_vitalSigns.contains(vitalSign)) {
        return;
    }

    auto& config = m_vitalSigns[vitalSign];
    config.data.append(qMakePair(timestamp, value));

    // Keep data sorted by time
    std::sort(config.data.begin(), config.data.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Remove old data outside time window
    QDateTime cutoffTime = QDateTime::currentDateTime().addSecs(-m_timeWindow * 3600);
    config.data.erase(
        std::remove_if(config.data.begin(), config.data.end(),
                      [cutoffTime](const auto& p) { return p.first < cutoffTime; }),
        config.data.end());

    updateChart();
}

void TrendChartWidget::addDataPoints(const QString& vitalSign,
                                    const QVector<QPair<QDateTime, double>>& data) {
    for (const auto& point : data) {
        addDataPoint(vitalSign, point.first, point.second);
    }
}

void TrendChartWidget::clearData(const QString& vitalSign) {
    if (m_vitalSigns.contains(vitalSign)) {
        m_vitalSigns[vitalSign].data.clear();
        updateChart();
    }
}

void TrendChartWidget::clearAllData() {
    for (auto& config : m_vitalSigns) {
        config.data.clear();
    }
    updateChart();
}

void TrendChartWidget::setViewMode(ViewMode mode) {
    if (m_viewMode != mode) {
        m_viewMode = mode;
        switchView();
        emit viewModeChanged(mode);
    }
}

void TrendChartWidget::setTimeWindow(TimeWindow window) {
    if (m_timeWindow != window) {
        m_timeWindow = window;
        m_startTime = m_endTime.addSecs(-m_timeWindow * 3600);
        updateChart();
        emit timeWindowChanged(window);
    }
}

void TrendChartWidget::updateChart() {
    m_endTime = QDateTime::currentDateTime();
    m_startTime = m_endTime.addSecs(-m_timeWindow * 3600);

    m_axisX->setRange(m_startTime, m_endTime);

    // Update all series
    for (auto& config : m_vitalSigns) {
        config.series->clear();

        for (const auto& point : config.data) {
            if (point.first >= m_startTime && point.first <= m_endTime) {
                config.series->append(point.first.toMSecsSinceEpoch(), point.second);
            }
        }
    }
}

void TrendChartWidget::updateTable() {
    m_tableWidget->setRowCount(0);

    int row = 0;
    for (const auto& config : m_vitalSigns) {
        if (!config.visible) continue;

        for (const auto& point : config.data) {
            if (point.first >= m_startTime && point.first <= m_endTime) {
                m_tableWidget->insertRow(row);
                m_tableWidget->setItem(row, 0,
                    new QTableWidgetItem(point.first.toString("yyyy-MM-dd hh:mm:ss")));
                m_tableWidget->setItem(row, 1,
                    new QTableWidgetItem(config.name));
                m_tableWidget->setItem(row, 2,
                    new QTableWidgetItem(QString::number(point.second, 'f', 1) +
                                        " " + config.unit));
                row++;
            }
        }
    }
}

void TrendChartWidget::updateHistogram() {
    // Clear existing series
    m_chart->removeAllSeries();

    // Create histogram for each vital sign
    for (auto& config : m_vitalSigns) {
        if (!config.visible || config.data.isEmpty()) continue;

        // Calculate histogram bins
        const int numBins = 20;
        QVector<int> bins(numBins, 0);
        double binWidth = (config.maxValue - config.minValue) / numBins;

        for (const auto& point : config.data) {
            if (point.first >= m_startTime && point.first <= m_endTime) {
                int binIndex = qBound(0, static_cast<int>((point.second - config.minValue) / binWidth), numBins - 1);
                bins[binIndex]++;
            }
        }

        // Create bar series
        auto* barSet = new QBarSet(config.name);
        barSet->setColor(config.color);
        for (int count : bins) {
            *barSet << count;
        }

        auto* barSeries = new QBarSeries();
        barSeries->append(barSet);
        m_chart->addSeries(barSeries);
    }
}

void TrendChartWidget::updateScatterPlot() {
    // Clear existing series
    m_chart->removeAllSeries();

    // Create scatter plot for each vital sign
    for (auto& config : m_vitalSigns) {
        if (!config.visible || config.data.isEmpty()) continue;

        auto* scatterSeries = new QScatterSeries();
        scatterSeries->setName(config.name);
        scatterSeries->setColor(config.color);
        scatterSeries->setMarkerSize(8.0);

        for (const auto& point : config.data) {
            if (point.first >= m_startTime && point.first <= m_endTime) {
                scatterSeries->append(point.first.toMSecsSinceEpoch(), point.second);
            }
        }

        m_chart->addSeries(scatterSeries);
        scatterSeries->attachAxis(m_axisX);
        scatterSeries->attachAxis(m_axisYMap[config.name]);
    }
}

void TrendChartWidget::switchView() {
    switch (m_viewMode) {
    case LineChart:
        m_chartView->show();
        m_tableWidget->hide();
        updateChart();
        break;

    case TabularView:
        m_chartView->hide();
        m_tableWidget->show();
        updateTable();
        break;

    case Histogram:
        m_chartView->show();
        m_tableWidget->hide();
        updateHistogram();
        break;

    case ScatterPlot:
        m_chartView->show();
        m_tableWidget->hide();
        updateScatterPlot();
        break;
    }
}

void TrendChartWidget::exportToCSV(const QString& filename) {
    QString file = filename;
    if (file.isEmpty()) {
        file = QFileDialog::getSaveFileName(this, "Export to CSV",
                                           QString(), "CSV Files (*.csv)");
    }

    if (!file.isEmpty()) {
        QFile csvFile(file);
        if (csvFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&csvFile);
            out << "Timestamp,Vital Sign,Value,Unit\n";

            for (const auto& config : m_vitalSigns) {
                for (const auto& point : config.data) {
                    if (point.first >= m_startTime && point.first <= m_endTime) {
                        out << point.first.toString("yyyy-MM-dd hh:mm:ss") << ","
                            << config.name << ","
                            << point.second << ","
                            << config.unit << "\n";
                    }
                }
            }

            csvFile.close();
            QMessageBox::information(this, "Export Complete",
                                    "Data exported to CSV successfully.");
        }
    }
}

void TrendChartWidget::exportToPDF(const QString& filename) {
    QString file = filename;
    if (file.isEmpty()) {
        file = QFileDialog::getSaveFileName(this, "Export to PDF",
                                           QString(), "PDF Files (*.pdf)");
    }

    if (!file.isEmpty()) {
        QPrinter printer(QPrinter::HighResolution);
        printer.setOutputFormat(QPrinter::PdfFormat);
        printer.setOutputFileName(file);
        printer.setPageSize(QPageSize::A4);
        printer.setPageOrientation(QPageLayout::Landscape);

        QPainter painter(&printer);
        m_chartView->render(&painter);

        QMessageBox::information(this, "Export Complete",
                                "Chart exported to PDF successfully.");
    }
}

QImage TrendChartWidget::captureChart() {
    return m_chartView->grab().toImage();
}

void TrendChartWidget::onTimeWindowChanged(int index) {
    TimeWindow window = static_cast<TimeWindow>(
        m_timeWindowCombo->itemData(index).toInt());
    setTimeWindow(window);
}

void TrendChartWidget::onViewModeChanged(int index) {
    ViewMode mode = static_cast<ViewMode>(
        m_viewModeCombo->itemData(index).toInt());
    setViewMode(mode);
}

void TrendChartWidget::onExportCSVClicked() {
    exportToCSV();
}

void TrendChartWidget::onExportPDFClicked() {
    exportToPDF();
}

void TrendChartWidget::onVitalSignToggled(const QString& name, bool checked) {
    setVitalSignVisible(name, checked);
}

} // namespace VitalStream
