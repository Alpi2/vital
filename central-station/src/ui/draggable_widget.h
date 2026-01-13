#ifndef DRAGGABLE_WIDGET_H
#define DRAGGABLE_WIDGET_H

#include <QWidget>
#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QMouseEvent>
#include <QDrag>
#include <QMimeData>
#include <QJsonObject>

namespace VitalStream {

/**
 * @brief Draggable widget for dashboard customization
 * 
 * Features:
 * - Drag and drop support
 * - Resize handles
 * - Widget configuration
 * - Save/restore layout
 */
class DraggableWidget : public QFrame {
    Q_OBJECT

public:
    enum WidgetType {
        WaveformWidget,
        VitalSignsWidget,
        TrendChartWidget,
        AlarmListWidget,
        PatientInfoWidget,
        ECG12LeadWidget,
        CustomWidget
    };

    explicit DraggableWidget(WidgetType type, QWidget* parent = nullptr);
    ~DraggableWidget() override;

    // Widget properties
    void setWidgetTitle(const QString& title);
    QString widgetTitle() const { return m_title; }
    WidgetType widgetType() const { return m_type; }
    void setWidgetID(const QString& id) { m_widgetID = id; }
    QString widgetID() const { return m_widgetID; }

    // Content widget
    void setContentWidget(QWidget* widget);
    QWidget* contentWidget() const { return m_contentWidget; }

    // Dragging
    void setDraggable(bool draggable) { m_draggable = draggable; }
    bool isDraggable() const { return m_draggable; }

    // Resizing
    void setResizable(bool resizable) { m_resizable = resizable; }
    bool isResizable() const { return m_resizable; }

    // Serialization
    QJsonObject toJson() const;
    void fromJson(const QJsonObject& json);

signals:
    void closeRequested();
    void configureRequested();
    void widgetMoved(const QPoint& newPos);
    void widgetResized(const QSize& newSize);

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void paintEvent(QPaintEvent* event) override;

private slots:
    void onCloseClicked();
    void onConfigureClicked();

private:
    void setupUI();
    void startDrag();

    WidgetType m_type;
    QString m_widgetID;
    QString m_title;
    bool m_draggable;
    bool m_resizable;

    // UI components
    QVBoxLayout* m_mainLayout;
    QWidget* m_titleBar;
    QLabel* m_titleLabel;
    QPushButton* m_configureButton;
    QPushButton* m_closeButton;
    QWidget* m_contentWidget;

    // Drag state
    bool m_dragging;
    QPoint m_dragStartPos;
};

/**
 * @brief Dashboard container for draggable widgets
 */
class DashboardContainer : public QWidget {
    Q_OBJECT

public:
    explicit DashboardContainer(QWidget* parent = nullptr);
    ~DashboardContainer() override;

    // Widget management
    void addWidget(DraggableWidget* widget, const QPoint& pos);
    void removeWidget(DraggableWidget* widget);
    void clearWidgets();
    QVector<DraggableWidget*> widgets() const { return m_widgets; }

    // Layout management
    void setGridSize(int rows, int cols);
    void enableSnapToGrid(bool enable) { m_snapToGrid = enable; }
    bool isSnapToGridEnabled() const { return m_snapToGrid; }

    // Serialization
    QJsonObject saveLayout() const;
    void loadLayout(const QJsonObject& json);

signals:
    void widgetAdded(DraggableWidget* widget);
    void widgetRemoved(DraggableWidget* widget);
    void layoutChanged();

protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dragMoveEvent(QDragMoveEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    void paintEvent(QPaintEvent* event) override;

private:
    QPoint snapToGrid(const QPoint& pos) const;

    QVector<DraggableWidget*> m_widgets;
    bool m_snapToGrid;
    int m_gridRows;
    int m_gridCols;
    int m_gridCellWidth;
    int m_gridCellHeight;
};

} // namespace VitalStream

#endif // DRAGGABLE_WIDGET_H
