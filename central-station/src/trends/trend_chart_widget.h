#ifndef TREND_CHART_WIDGET_H
#define TREND_CHART_WIDGET_H

#include <QWidget>
#include <QChart>
#include <QChartView>
#include <QLineSeries>
#include <QDateTimeAxis>
#include <QValueAxis>
#include <QComboBox>
#include <QPushButton>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QMap>
#include <QDateTime>

QT_CHARTS_USE_NAMESPACE

namespace VitalStream {

/**
 * @brief Trend chart widget for vital signs over time
 * 
 * Features:
 * - Multiple time windows (1h, 6h, 12h, 24h, 48h, 72h)
 * - Multiple vital signs on same chart
 * - Tabular view
 * - Histogram and scatter plot
 * - Export to CSV/PDF
 * - Zoom and pan
 */
class TrendChartWidget : public QWidget {
    Q_OBJECT

public:
    enum ViewMode {
        LineChart,      // Line chart (default)
        TabularView,    // Table view
        Histogram,      // Histogram
        ScatterPlot     // Scatter plot
    };

    enum TimeWindow {
        OneHour = 1,
        SixHours = 6,
        TwelveHours = 12,
        TwentyFourHours = 24,
        FortyEightHours = 48,
        SeventyTwoHours = 72
    };

    explicit TrendChartWidget(QWidget* parent = nullptr);
    ~TrendChartWidget() override;

    // Data management
    void addDataPoint(const QString& vitalSign, const QDateTime& timestamp, double value);
    void addDataPoints(const QString& vitalSign,
                      const QVector<QPair<QDateTime, double>>& data);
    void clearData(const QString& vitalSign);
    void clearAllData();

    // Vital sign configuration
    void addVitalSign(const QString& name, const QColor& color,
                     double minValue, double maxValue, const QString& unit);
    void removeVitalSign(const QString& name);
    void setVitalSignVisible(const QString& name, bool visible);

    // View control
    void setViewMode(ViewMode mode);
    ViewMode viewMode() const { return m_viewMode; }
    void setTimeWindow(TimeWindow window);
    TimeWindow timeWindow() const { return m_timeWindow; }

    // Export
    void exportToCSV(const QString& filename);
    void exportToPDF(const QString& filename);
    QImage captureChart();

signals:
    void dataPointClicked(const QString& vitalSign, const QDateTime& timestamp, double value);
    void timeWindowChanged(TimeWindow window);
    void viewModeChanged(ViewMode mode);

private slots:
    void onTimeWindowChanged(int index);
    void onViewModeChanged(int index);
    void onExportCSVClicked();
    void onExportPDFClicked();
    void onVitalSignToggled(const QString& name, bool checked);

private:
    void setupUI();
    void setupChart();
    void setupTable();
    void updateChart();
    void updateTable();
    void updateHistogram();
    void updateScatterPlot();
    void switchView();

    struct VitalSignConfig {
        QString name;
        QColor color;
        double minValue;
        double maxValue;
        QString unit;
        bool visible;
        QLineSeries* series;
        QVector<QPair<QDateTime, double>> data;
    };

    ViewMode m_viewMode;
    TimeWindow m_timeWindow;

    // UI components
    QVBoxLayout* m_mainLayout;
    QWidget* m_controlPanel;
    QComboBox* m_timeWindowCombo;
    QComboBox* m_viewModeCombo;
    QPushButton* m_exportCSVButton;
    QPushButton* m_exportPDFButton;

    // Chart view
    QChart* m_chart;
    QChartView* m_chartView;
    QDateTimeAxis* m_axisX;
    QMap<QString, QValueAxis*> m_axisYMap;

    // Table view
    QTableWidget* m_tableWidget;

    // Data
    QMap<QString, VitalSignConfig> m_vitalSigns;

    // Time range
    QDateTime m_startTime;
    QDateTime m_endTime;
};

} // namespace VitalStream

#endif // TREND_CHART_WIDGET_H
