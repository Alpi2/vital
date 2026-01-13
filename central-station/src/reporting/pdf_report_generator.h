#ifndef PDF_REPORT_GENERATOR_H
#define PDF_REPORT_GENERATOR_H

#include <QObject>
#include <QString>
#include <QDateTime>
#include <QPrinter>
#include <QPainter>
#include <QImage>
#include <QVector>
#include <QMap>

namespace VitalStream {

/**
 * @brief Report data structure
 */
struct ReportData {
    // Patient information
    QString patientName;
    QString patientID;
    QString mrn;
    QDateTime dateOfBirth;
    QString gender;
    QString department;
    QString room;
    QString bed;
    
    // Report metadata
    QString reportType;
    QDateTime reportDate;
    QDateTime startTime;
    QDateTime endTime;
    QString generatedBy;
    QString institution;
    
    // Vital signs data
    struct VitalSignData {
        QString name;
        QVector<QPair<QDateTime, double>> values;
        QString unit;
        double minValue;
        double maxValue;
        double avgValue;
    };
    QVector<VitalSignData> vitalSigns;
    
    // ECG data
    struct ECGData {
        QMap<QString, QVector<float>> leads; // 12 leads
        int heartRate;
        int prInterval;
        int qrsDuration;
        int qtInterval;
        int qtcInterval;
        int axis;
        QString interpretation;
        QStringList findings;
    };
    ECGData ecgData;
    
    // Alarms
    struct AlarmData {
        QDateTime timestamp;
        QString type;
        QString severity;
        QString parameter;
        double value;
        QString action;
    };
    QVector<AlarmData> alarms;
    
    // Images (waveforms, charts, etc.)
    QVector<QImage> images;
    QVector<QString> imageLabels;
    
    // Clinical notes
    QString clinicalNotes;
    QString diagnosis;
    QString treatment;
};

/**
 * @brief PDF report generator
 * 
 * Features:
 * - Patient summary reports
 * - ECG reports
 * - Trend reports
 * - Alarm reports
 * - Custom templates
 * - Multi-page support
 * - Headers and footers
 * - Page numbers
 */
class PDFReportGenerator : public QObject {
    Q_OBJECT

public:
    enum ReportType {
        PatientSummary,
        ECGReport,
        TrendReport,
        AlarmReport,
        ComprehensiveReport,
        CustomReport
    };

    enum PageSize {
        A4,
        Letter,
        Legal
    };

    enum Orientation {
        Portrait,
        Landscape
    };

    explicit PDFReportGenerator(QObject* parent = nullptr);
    ~PDFReportGenerator() override;

    // Report generation
    bool generateReport(const ReportData& data, const QString& filename,
                       ReportType type = PatientSummary);
    bool generatePatientSummary(const ReportData& data, const QString& filename);
    bool generateECGReport(const ReportData& data, const QString& filename);
    bool generateTrendReport(const ReportData& data, const QString& filename);
    bool generateAlarmReport(const ReportData& data, const QString& filename);

    // Configuration
    void setPageSize(PageSize size);
    void setOrientation(Orientation orientation);
    void setMargins(int left, int top, int right, int bottom);
    void setInstitutionLogo(const QImage& logo);
    void setInstitutionName(const QString& name);
    void setHeaderText(const QString& text);
    void setFooterText(const QString& text);
    void setIncludePageNumbers(bool include) { m_includePageNumbers = include; }
    void setIncludeTimestamp(bool include) { m_includeTimestamp = include; }

    // Preview
    QImage generatePreview(const ReportData& data, ReportType type);

signals:
    void reportGenerated(const QString& filename);
    void reportGenerationFailed(const QString& error);
    void progressChanged(int percentage);

private:
    void setupPrinter(QPrinter& printer);
    void drawHeader(QPainter& painter, const QRectF& rect, const ReportData& data);
    void drawFooter(QPainter& painter, const QRectF& rect, int pageNumber, int totalPages);
    
    // Report sections
    void drawPatientInfo(QPainter& painter, QRectF& rect, const ReportData& data);
    void drawVitalSignsTable(QPainter& painter, QRectF& rect, const ReportData& data);
    void drawVitalSignsChart(QPainter& painter, QRectF& rect, const ReportData& data);
    void drawECGWaveforms(QPainter& painter, QRectF& rect, const ReportData& data);
    void drawECGMeasurements(QPainter& painter, QRectF& rect, const ReportData& data);
    void drawAlarmList(QPainter& painter, QRectF& rect, const ReportData& data);
    void drawClinicalNotes(QPainter& painter, QRectF& rect, const ReportData& data);
    void drawImage(QPainter& painter, QRectF& rect, const QImage& image, const QString& label);
    
    // Helper functions
    QRectF getContentRect(const QPrinter& printer) const;
    void drawText(QPainter& painter, const QRectF& rect, const QString& text,
                 Qt::Alignment alignment = Qt::AlignLeft | Qt::AlignTop);
    void drawTable(QPainter& painter, QRectF& rect,
                  const QStringList& headers,
                  const QVector<QStringList>& rows);
    qreal calculateTextHeight(const QString& text, const QFont& font, qreal width) const;
    bool needsNewPage(const QRectF& rect, qreal requiredHeight) const;
    void newPage(QPrinter& printer, QPainter& painter, QRectF& rect);

    PageSize m_pageSize;
    Orientation m_orientation;
    int m_marginLeft;
    int m_marginTop;
    int m_marginRight;
    int m_marginBottom;
    
    QImage m_institutionLogo;
    QString m_institutionName;
    QString m_headerText;
    QString m_footerText;
    bool m_includePageNumbers;
    bool m_includeTimestamp;
    
    int m_currentPage;
    int m_totalPages;
};

/**
 * @brief Report template manager
 */
class ReportTemplateManager : public QObject {
    Q_OBJECT

public:
    explicit ReportTemplateManager(QObject* parent = nullptr);

    // Template management
    bool loadTemplate(const QString& name, const QString& filename);
    bool saveTemplate(const QString& name, const QString& filename);
    QStringList availableTemplates() const;
    
    // Apply template
    void applyTemplate(PDFReportGenerator* generator, const QString& templateName);

private:
    struct ReportTemplate {
        QString name;
        PDFReportGenerator::PageSize pageSize;
        PDFReportGenerator::Orientation orientation;
        int marginLeft, marginTop, marginRight, marginBottom;
        QString headerText;
        QString footerText;
        bool includePageNumbers;
        bool includeTimestamp;
    };
    
    QMap<QString, ReportTemplate> m_templates;
};

} // namespace VitalStream

#endif // PDF_REPORT_GENERATOR_H
