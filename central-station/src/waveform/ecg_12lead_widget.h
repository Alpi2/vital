#ifndef ECG_12LEAD_WIDGET_H
#define ECG_12LEAD_WIDGET_H

#include "waveform_renderer.h"
#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>

namespace VitalStream {

/**
 * @brief 12-lead ECG display widget
 * 
 * Supports multiple display formats:
 * - Standard 3x4 grid (3 columns, 4 rows + rhythm strip)
 * - Cabrera format (anatomical ordering)
 * - 6x2 format
 * - Single lead full-screen
 */
class ECG12LeadWidget : public QWidget {
    Q_OBJECT

public:
    enum DisplayFormat {
        Standard_3x4,      // Standard 3x4 grid
        Cabrera,           // Cabrera format (aVL, I, -aVR, II, aVF, III, V1-V6)
        Format_6x2,        // 6 columns, 2 rows
        SingleLead,        // Single lead full-screen
        Rhythm_Strip       // Long rhythm strip (Lead II)
    };

    explicit ECG12LeadWidget(QWidget* parent = nullptr);
    ~ECG12LeadWidget() override;

    // Data input
    void updateECGData(const QString& leadName, const QVector<float>& data);
    void updateAllLeads(const QMap<QString, QVector<float>>& allLeadsData);

    // Display format
    void setDisplayFormat(DisplayFormat format);
    DisplayFormat displayFormat() const { return m_displayFormat; }

    // Measurements and annotations
    void setHeartRate(int bpm);
    void setPRInterval(int ms);
    void setQRSDuration(int ms);
    void setQTInterval(int ms);
    void setQTcInterval(int ms);
    void setAxis(int degrees);

    // Interpretation
    void setInterpretation(const QString& text);
    void addFinding(const QString& finding);
    void clearFindings();

    // Print and export
    void printECG();
    QImage captureECG();
    void exportToPDF(const QString& filename);

signals:
    void leadSelected(const QString& leadName);
    void formatChanged(DisplayFormat format);

private slots:
    void onFormatChanged(int index);
    void onPrintClicked();
    void onExportClicked();

private:
    void setupUI();
    void setupLeadRenderers();
    void arrangeLeads();
    void updateMeasurementLabels();

    // Lead names in different formats
    static QStringList standardLeadOrder();
    static QStringList cabreraLeadOrder();

    DisplayFormat m_displayFormat;

    // Waveform renderers for each lead
    QMap<QString, WaveformRenderer*> m_leadRenderers;

    // UI components
    QVBoxLayout* m_mainLayout;
    QWidget* m_headerWidget;
    QWidget* m_waveformContainer;
    QWidget* m_measurementWidget;
    QWidget* m_interpretationWidget;
    QComboBox* m_formatComboBox;
    QPushButton* m_printButton;
    QPushButton* m_exportButton;

    // Measurements
    QLabel* m_heartRateLabel;
    QLabel* m_prIntervalLabel;
    QLabel* m_qrsDurationLabel;
    QLabel* m_qtIntervalLabel;
    QLabel* m_qtcIntervalLabel;
    QLabel* m_axisLabel;

    // Interpretation
    QLabel* m_interpretationLabel;
    QStringList m_findings;

    // Patient info (optional)
    QString m_patientName;
    QString m_patientID;
    QDateTime m_acquisitionTime;
};

} // namespace VitalStream

#endif // ECG_12LEAD_WIDGET_H
