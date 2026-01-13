#ifndef TOUCHSCREEN_SUPPORT_H
#define TOUCHSCREEN_SUPPORT_H

#include <QWidget>
#include <QGestureEvent>
#include <QPinchGesture>
#include <QSwipeGesture>
#include <QTapGesture>
#include <QScroller>

namespace VitalStream {

/**
 * @brief Touchscreen support mixin for widgets
 * 
 * Features:
 * - Pinch to zoom
 * - Swipe gestures
 * - Tap and long press
 * - Kinetic scrolling
 * - Large touch targets
 */
class TouchscreenSupport {
public:
    TouchscreenSupport();
    virtual ~TouchscreenSupport();

    // Enable/disable touch support
    void enableTouchSupport(QWidget* widget);
    void disableTouchSupport(QWidget* widget);

    // Gesture configuration
    void setPinchZoomEnabled(bool enabled) { m_pinchZoomEnabled = enabled; }
    bool isPinchZoomEnabled() const { return m_pinchZoomEnabled; }
    
    void setSwipeEnabled(bool enabled) { m_swipeEnabled = enabled; }
    bool isSwipeEnabled() const { return m_swipeEnabled; }
    
    void setKineticScrollingEnabled(bool enabled) { m_kineticScrollingEnabled = enabled; }
    bool isKineticScrollingEnabled() const { return m_kineticScrollingEnabled; }

    // Touch target size
    static int minimumTouchTargetSize() { return 44; } // 44px minimum (iOS HIG)
    static int recommendedTouchTargetSize() { return 48; } // 48dp (Material Design)

protected:
    // Gesture handlers (to be called from widget's event handler)
    bool handleGestureEvent(QGestureEvent* event);
    bool handlePinchGesture(QPinchGesture* gesture);
    bool handleSwipeGesture(QSwipeGesture* gesture);
    bool handleTapGesture(QTapGesture* gesture);

    // Virtual callbacks for derived classes
    virtual void onPinchZoom(qreal scaleFactor, const QPointF& center) {}
    virtual void onSwipeLeft() {}
    virtual void onSwipeRight() {}
    virtual void onSwipeUp() {}
    virtual void onSwipeDown() {}
    virtual void onTap(const QPointF& pos) {}
    virtual void onLongPress(const QPointF& pos) {}

private:
    bool m_pinchZoomEnabled;
    bool m_swipeEnabled;
    bool m_kineticScrollingEnabled;
    qreal m_currentScale;
};

/**
 * @brief Touchscreen-optimized button
 */
class TouchButton : public QPushButton {
    Q_OBJECT

public:
    explicit TouchButton(const QString& text, QWidget* parent = nullptr);
    explicit TouchButton(const QIcon& icon, const QString& text, QWidget* parent = nullptr);

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

protected:
    void paintEvent(QPaintEvent* event) override;
};

/**
 * @brief Touchscreen-optimized scroll area
 */
class TouchScrollArea : public QScrollArea, public TouchscreenSupport {
    Q_OBJECT

public:
    explicit TouchScrollArea(QWidget* parent = nullptr);

protected:
    bool event(QEvent* event) override;
    bool gestureEvent(QGestureEvent* event);
};

} // namespace VitalStream

#endif // TOUCHSCREEN_SUPPORT_H
