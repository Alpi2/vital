#ifndef WAVEFORM_RENDERER_H
#define WAVEFORM_RENDERER_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>
#include <QTimer>
#include <QVector>
#include <QColor>
#include <deque>
#include <memory>

namespace VitalStream {

/**
 * @brief Waveform channel data structure
 */
struct WaveformChannel {
    QString name;              // Channel name (e.g., "ECG I", "ECG II")
    QColor color;              // Waveform color
    std::deque<float> data;    // Circular buffer for waveform data
    float yOffset;             // Vertical offset for multi-channel display
    float scale;               // Amplitude scale
    bool visible;              // Channel visibility
    float minValue;            // Min value for auto-scaling
    float maxValue;            // Max value for auto-scaling
    
    WaveformChannel(const QString& n = "", const QColor& c = Qt::green)
        : name(n), color(c), yOffset(0.0f), scale(1.0f), visible(true),
          minValue(-1.0f), maxValue(1.0f) {}
};

/**
 * @brief High-performance OpenGL waveform renderer
 * 
 * Features:
 * - 60+ FPS rendering for 6+ channels
 * - Real-time data streaming (1000+ samples/sec per channel)
 * - Freeze/unfreeze capability
 * - Zoom and pan
 * - Cascade and overlay modes
 * - Grid and annotations
 * - Hardware-accelerated rendering
 */
class WaveformRenderer : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT

public:
    enum DisplayMode {
        Cascade,    // Each channel in separate row
        Overlay,    // All channels overlaid
        Grid        // Grid layout (2x3, 3x4, etc.)
    };

    explicit WaveformRenderer(QWidget* parent = nullptr);
    ~WaveformRenderer() override;

    // Channel management
    void addChannel(const QString& name, const QColor& color = Qt::green);
    void removeChannel(int channelIndex);
    void setChannelVisible(int channelIndex, bool visible);
    void setChannelColor(int channelIndex, const QColor& color);
    void setChannelScale(int channelIndex, float scale);
    int channelCount() const { return m_channels.size(); }

    // Data management
    void addDataPoint(int channelIndex, float value);
    void addDataPoints(int channelIndex, const QVector<float>& values);
    void clearChannel(int channelIndex);
    void clearAllChannels();

    // Display control
    void setDisplayMode(DisplayMode mode);
    DisplayMode displayMode() const { return m_displayMode; }
    void setTimeWindow(float seconds); // Time window in seconds
    float timeWindow() const { return m_timeWindow; }
    void setSampleRate(float hz); // Sample rate in Hz
    float sampleRate() const { return m_sampleRate; }

    // Freeze/unfreeze
    void freeze();
    void unfreeze();
    bool isFrozen() const { return m_frozen; }
    void toggleFreeze();

    // Zoom and pan
    void zoomIn();
    void zoomOut();
    void setZoomLevel(float level); // 1.0 = normal, 2.0 = 2x zoom
    float zoomLevel() const { return m_zoomLevel; }
    void panLeft();
    void panRight();
    void panUp();
    void panDown();
    void resetView();

    // Grid and annotations
    void setGridVisible(bool visible);
    bool isGridVisible() const { return m_showGrid; }
    void setGridColor(const QColor& color);
    void setBackgroundColor(const QColor& color);

    // Screenshot
    QImage captureWaveform();

    // Performance metrics
    float currentFPS() const { return m_currentFPS; }
    int totalDataPoints() const;

signals:
    void fpsChanged(float fps);
    void frozenStateChanged(bool frozen);
    void zoomLevelChanged(float level);

protected:
    // OpenGL overrides
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    // Event handlers
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private slots:
    void updateFrame();
    void calculateFPS();

private:
    void setupShaders();
    void setupBuffers();
    void updateBuffers();
    void drawGrid();
    void drawWaveforms();
    void drawAnnotations();
    void calculateChannelOffsets();

    // Channels
    QVector<WaveformChannel> m_channels;
    DisplayMode m_displayMode;

    // Timing
    float m_timeWindow;        // Time window in seconds (default: 10s)
    float m_sampleRate;        // Sample rate in Hz (default: 250 Hz)
    int m_maxDataPoints;       // Max data points per channel

    // State
    bool m_frozen;
    float m_zoomLevel;
    float m_panX;
    float m_panY;

    // Grid and colors
    bool m_showGrid;
    QColor m_gridColor;
    QColor m_backgroundColor;

    // OpenGL resources
    std::unique_ptr<QOpenGLShaderProgram> m_shaderProgram;
    QOpenGLBuffer m_vertexBuffer;
    QOpenGLVertexArrayObject m_vao;

    // Rendering
    QTimer* m_renderTimer;
    QTimer* m_fpsTimer;
    float m_currentFPS;
    int m_frameCount;

    // Mouse interaction
    bool m_mousePressed;
    QPoint m_lastMousePos;

    // Performance optimization
    bool m_needsBufferUpdate;
    QVector<float> m_vertexData;
};

} // namespace VitalStream

#endif // WAVEFORM_RENDERER_H
