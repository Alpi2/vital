#include "waveform_renderer.h"
#include <QMouseEvent>
#include <QWheelEvent>
#include <QPainter>
#include <QOpenGLContext>
#include <QtMath>
#include <algorithm>

namespace VitalStream {

// Vertex shader for waveform rendering
static const char* vertexShaderSource = R"(
    #version 330 core
    layout(location = 0) in vec2 position;
    layout(location = 1) in vec3 color;
    
    uniform mat4 projection;
    uniform mat4 view;
    
    out vec3 fragColor;
    
    void main() {
        gl_Position = projection * view * vec4(position, 0.0, 1.0);
        fragColor = color;
    }
)";

// Fragment shader
static const char* fragmentShaderSource = R"(
    #version 330 core
    in vec3 fragColor;
    out vec4 outColor;
    
    void main() {
        outColor = vec4(fragColor, 1.0);
    }
)";

WaveformRenderer::WaveformRenderer(QWidget* parent)
    : QOpenGLWidget(parent)
    , m_displayMode(DisplayMode::Cascade)
    , m_timeWindow(10.0f)  // 10 seconds default
    , m_sampleRate(250.0f) // 250 Hz default
    , m_frozen(false)
    , m_zoomLevel(1.0f)
    , m_panX(0.0f)
    , m_panY(0.0f)
    , m_showGrid(true)
    , m_gridColor(60, 60, 60)
    , m_backgroundColor(0, 0, 0)
    , m_vertexBuffer(QOpenGLBuffer::VertexBuffer)
    , m_currentFPS(0.0f)
    , m_frameCount(0)
    , m_mousePressed(false)
    , m_needsBufferUpdate(true)
{
    // Calculate max data points
    m_maxDataPoints = static_cast<int>(m_timeWindow * m_sampleRate);

    // Setup render timer for 60 FPS
    m_renderTimer = new QTimer(this);
    connect(m_renderTimer, &QTimer::timeout, this, &WaveformRenderer::updateFrame);
    m_renderTimer->start(16); // ~60 FPS (16ms)

    // Setup FPS calculation timer
    m_fpsTimer = new QTimer(this);
    connect(m_fpsTimer, &QTimer::timeout, this, &WaveformRenderer::calculateFPS);
    m_fpsTimer->start(1000); // Update FPS every second

    // Enable mouse tracking for pan
    setMouseTracking(true);
}

WaveformRenderer::~WaveformRenderer() {
    makeCurrent();
    m_vertexBuffer.destroy();
    m_vao.destroy();
    doneCurrent();
}

void WaveformRenderer::initializeGL() {
    initializeOpenGLFunctions();

    // Set background color
    glClearColor(m_backgroundColor.redF(), m_backgroundColor.greenF(),
                 m_backgroundColor.blueF(), 1.0f);

    // Enable blending for smooth lines
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Enable line smoothing
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    // Setup shaders
    setupShaders();

    // Setup buffers
    setupBuffers();
}

void WaveformRenderer::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
    calculateChannelOffsets();
}

void WaveformRenderer::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT);

    if (!m_shaderProgram || !m_shaderProgram->bind()) {
        return;
    }

    // Update projection matrix
    QMatrix4x4 projection;
    projection.ortho(0.0f, static_cast<float>(width()),
                     static_cast<float>(height()), 0.0f,
                     -1.0f, 1.0f);

    // Update view matrix (for zoom and pan)
    QMatrix4x4 view;
    view.translate(m_panX, m_panY);
    view.scale(m_zoomLevel, m_zoomLevel);

    m_shaderProgram->setUniformValue("projection", projection);
    m_shaderProgram->setUniformValue("view", view);

    // Draw grid
    if (m_showGrid) {
        drawGrid();
    }

    // Draw waveforms
    drawWaveforms();

    // Draw annotations
    drawAnnotations();

    m_shaderProgram->release();

    m_frameCount++;
}

void WaveformRenderer::setupShaders() {
    m_shaderProgram = std::make_unique<QOpenGLShaderProgram>();

    if (!m_shaderProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource)) {
        qWarning() << "Failed to compile vertex shader:" << m_shaderProgram->log();
        return;
    }

    if (!m_shaderProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource)) {
        qWarning() << "Failed to compile fragment shader:" << m_shaderProgram->log();
        return;
    }

    if (!m_shaderProgram->link()) {
        qWarning() << "Failed to link shader program:" << m_shaderProgram->log();
        return;
    }
}

void WaveformRenderer::setupBuffers() {
    m_vao.create();
    m_vao.bind();

    m_vertexBuffer.create();
    m_vertexBuffer.bind();
    m_vertexBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);

    // Setup vertex attributes (position + color)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                          reinterpret_cast<void*>(2 * sizeof(float)));

    m_vertexBuffer.release();
    m_vao.release();
}

void WaveformRenderer::updateBuffers() {
    if (!m_needsBufferUpdate) {
        return;
    }

    m_vertexData.clear();

    // Build vertex data for all visible channels
    for (int ch = 0; ch < m_channels.size(); ++ch) {
        const auto& channel = m_channels[ch];
        if (!channel.visible || channel.data.empty()) {
            continue;
        }

        const float channelHeight = height() / static_cast<float>(m_channels.size());
        const float baseY = ch * channelHeight + channelHeight / 2.0f + channel.yOffset;
        const float xStep = width() / static_cast<float>(m_maxDataPoints);

        // Convert data points to vertices
        for (size_t i = 0; i < channel.data.size(); ++i) {
            float x = i * xStep;
            float y = baseY - (channel.data[i] * channel.scale * channelHeight * 0.4f);

            // Position (x, y)
            m_vertexData.append(x);
            m_vertexData.append(y);

            // Color (r, g, b)
            m_vertexData.append(channel.color.redF());
            m_vertexData.append(channel.color.greenF());
            m_vertexData.append(channel.color.blueF());
        }
    }

    // Upload to GPU
    m_vertexBuffer.bind();
    m_vertexBuffer.allocate(m_vertexData.constData(),
                            m_vertexData.size() * sizeof(float));
    m_vertexBuffer.release();

    m_needsBufferUpdate = false;
}

void WaveformRenderer::drawGrid() {
    // Grid drawing implementation
    // Draw major and minor grid lines
    const int majorGridX = 50;  // pixels
    const int majorGridY = 50;
    const int minorGridX = 10;
    const int minorGridY = 10;

    glLineWidth(1.0f);

    // This would be implemented with separate grid vertex buffer
    // For now, placeholder
}

void WaveformRenderer::drawWaveforms() {
    if (m_vertexData.isEmpty()) {
        return;
    }

    m_vao.bind();

    // Draw each channel as line strip
    int vertexOffset = 0;
    for (const auto& channel : m_channels) {
        if (!channel.visible || channel.data.empty()) {
            continue;
        }

        glLineWidth(2.0f);
        glDrawArrays(GL_LINE_STRIP, vertexOffset, channel.data.size());
        vertexOffset += channel.data.size();
    }

    m_vao.release();
}

void WaveformRenderer::drawAnnotations() {
    // Draw channel labels, time markers, etc.
    // This would use QPainter overlay or texture-based text rendering
}

void WaveformRenderer::calculateChannelOffsets() {
    if (m_displayMode == DisplayMode::Cascade) {
        // Evenly distribute channels vertically
        for (int i = 0; i < m_channels.size(); ++i) {
            m_channels[i].yOffset = 0.0f; // Calculated in updateBuffers
        }
    } else if (m_displayMode == DisplayMode::Overlay) {
        // All channels at same position
        for (auto& channel : m_channels) {
            channel.yOffset = 0.0f;
        }
    }
}

// Channel management
void WaveformRenderer::addChannel(const QString& name, const QColor& color) {
    WaveformChannel channel(name, color);
    channel.data.resize(m_maxDataPoints, 0.0f);
    m_channels.append(channel);
    calculateChannelOffsets();
    m_needsBufferUpdate = true;
}

void WaveformRenderer::removeChannel(int channelIndex) {
    if (channelIndex >= 0 && channelIndex < m_channels.size()) {
        m_channels.removeAt(channelIndex);
        calculateChannelOffsets();
        m_needsBufferUpdate = true;
    }
}

void WaveformRenderer::setChannelVisible(int channelIndex, bool visible) {
    if (channelIndex >= 0 && channelIndex < m_channels.size()) {
        m_channels[channelIndex].visible = visible;
        m_needsBufferUpdate = true;
    }
}

void WaveformRenderer::setChannelColor(int channelIndex, const QColor& color) {
    if (channelIndex >= 0 && channelIndex < m_channels.size()) {
        m_channels[channelIndex].color = color;
        m_needsBufferUpdate = true;
    }
}

void WaveformRenderer::setChannelScale(int channelIndex, float scale) {
    if (channelIndex >= 0 && channelIndex < m_channels.size()) {
        m_channels[channelIndex].scale = scale;
        m_needsBufferUpdate = true;
    }
}

// Data management
void WaveformRenderer::addDataPoint(int channelIndex, float value) {
    if (channelIndex < 0 || channelIndex >= m_channels.size() || m_frozen) {
        return;
    }

    auto& channel = m_channels[channelIndex];

    // Add to circular buffer
    if (channel.data.size() >= static_cast<size_t>(m_maxDataPoints)) {
        channel.data.pop_front();
    }
    channel.data.push_back(value);

    // Update min/max for auto-scaling
    channel.minValue = std::min(channel.minValue, value);
    channel.maxValue = std::max(channel.maxValue, value);

    m_needsBufferUpdate = true;
}

void WaveformRenderer::addDataPoints(int channelIndex, const QVector<float>& values) {
    for (float value : values) {
        addDataPoint(channelIndex, value);
    }
}

void WaveformRenderer::clearChannel(int channelIndex) {
    if (channelIndex >= 0 && channelIndex < m_channels.size()) {
        m_channels[channelIndex].data.clear();
        m_channels[channelIndex].data.resize(m_maxDataPoints, 0.0f);
        m_needsBufferUpdate = true;
    }
}

void WaveformRenderer::clearAllChannels() {
    for (auto& channel : m_channels) {
        channel.data.clear();
        channel.data.resize(m_maxDataPoints, 0.0f);
    }
    m_needsBufferUpdate = true;
}

// Display control
void WaveformRenderer::setDisplayMode(DisplayMode mode) {
    m_displayMode = mode;
    calculateChannelOffsets();
    m_needsBufferUpdate = true;
    update();
}

void WaveformRenderer::setTimeWindow(float seconds) {
    m_timeWindow = seconds;
    m_maxDataPoints = static_cast<int>(m_timeWindow * m_sampleRate);

    // Resize all channel buffers
    for (auto& channel : m_channels) {
        channel.data.resize(m_maxDataPoints, 0.0f);
    }

    m_needsBufferUpdate = true;
}

void WaveformRenderer::setSampleRate(float hz) {
    m_sampleRate = hz;
    m_maxDataPoints = static_cast<int>(m_timeWindow * m_sampleRate);

    for (auto& channel : m_channels) {
        channel.data.resize(m_maxDataPoints, 0.0f);
    }

    m_needsBufferUpdate = true;
}

// Freeze/unfreeze
void WaveformRenderer::freeze() {
    if (!m_frozen) {
        m_frozen = true;
        emit frozenStateChanged(true);
    }
}

void WaveformRenderer::unfreeze() {
    if (m_frozen) {
        m_frozen = false;
        emit frozenStateChanged(false);
    }
}

void WaveformRenderer::toggleFreeze() {
    m_frozen = !m_frozen;
    emit frozenStateChanged(m_frozen);
}

// Zoom and pan
void WaveformRenderer::zoomIn() {
    setZoomLevel(m_zoomLevel * 1.2f);
}

void WaveformRenderer::zoomOut() {
    setZoomLevel(m_zoomLevel / 1.2f);
}

void WaveformRenderer::setZoomLevel(float level) {
    m_zoomLevel = qBound(0.1f, level, 10.0f);
    emit zoomLevelChanged(m_zoomLevel);
    update();
}

void WaveformRenderer::panLeft() {
    m_panX += 50.0f / m_zoomLevel;
    update();
}

void WaveformRenderer::panRight() {
    m_panX -= 50.0f / m_zoomLevel;
    update();
}

void WaveformRenderer::panUp() {
    m_panY += 50.0f / m_zoomLevel;
    update();
}

void WaveformRenderer::panDown() {
    m_panY -= 50.0f / m_zoomLevel;
    update();
}

void WaveformRenderer::resetView() {
    m_zoomLevel = 1.0f;
    m_panX = 0.0f;
    m_panY = 0.0f;
    emit zoomLevelChanged(m_zoomLevel);
    update();
}

// Grid and colors
void WaveformRenderer::setGridVisible(bool visible) {
    m_showGrid = visible;
    update();
}

void WaveformRenderer::setGridColor(const QColor& color) {
    m_gridColor = color;
    update();
}

void WaveformRenderer::setBackgroundColor(const QColor& color) {
    m_backgroundColor = color;
    makeCurrent();
    glClearColor(color.redF(), color.greenF(), color.blueF(), 1.0f);
    doneCurrent();
    update();
}

// Screenshot
QImage WaveformRenderer::captureWaveform() {
    return grabFramebuffer();
}

int WaveformRenderer::totalDataPoints() const {
    int total = 0;
    for (const auto& channel : m_channels) {
        total += channel.data.size();
    }
    return total;
}

// Event handlers
void WaveformRenderer::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        m_mousePressed = true;
        m_lastMousePos = event->pos();
    }
}

void WaveformRenderer::mouseMoveEvent(QMouseEvent* event) {
    if (m_mousePressed) {
        QPoint delta = event->pos() - m_lastMousePos;
        m_panX += delta.x();
        m_panY += delta.y();
        m_lastMousePos = event->pos();
        update();
    }
}

void WaveformRenderer::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        m_mousePressed = false;
    }
}

void WaveformRenderer::wheelEvent(QWheelEvent* event) {
    // Zoom with mouse wheel
    float delta = event->angleDelta().y() / 120.0f;
    if (delta > 0) {
        zoomIn();
    } else if (delta < 0) {
        zoomOut();
    }
}

// Slots
void WaveformRenderer::updateFrame() {
    if (m_needsBufferUpdate) {
        updateBuffers();
    }
    update();
}

void WaveformRenderer::calculateFPS() {
    m_currentFPS = m_frameCount;
    m_frameCount = 0;
    emit fpsChanged(m_currentFPS);
}

} // namespace VitalStream
