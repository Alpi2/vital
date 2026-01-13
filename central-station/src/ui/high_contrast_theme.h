#ifndef HIGH_CONTRAST_THEME_H
#define HIGH_CONTRAST_THEME_H

#include <QObject>
#include <QString>
#include <QColor>
#include <QPalette>
#include <QApplication>

namespace VitalStream {

/**
 * @brief High contrast theme manager for accessibility
 * 
 * Features:
 * - High contrast color schemes
 * - Large fonts
 * - Clear visual indicators
 * - WCAG 2.1 AA compliance
 */
class HighContrastTheme : public QObject {
    Q_OBJECT

public:
    enum ContrastMode {
        Normal,           // Standard theme
        HighContrast,     // High contrast (black on white)
        HighContrastDark, // High contrast dark (white on black)
        YellowOnBlack,    // Yellow on black (classic)
        WhiteOnBlue       // White on blue
    };

    explicit HighContrastTheme(QObject* parent = nullptr);
    ~HighContrastTheme() override;

    // Theme management
    void setContrastMode(ContrastMode mode);
    ContrastMode contrastMode() const { return m_currentMode; }

    // Apply theme
    void applyTheme(QApplication* app);
    void applyTheme(QWidget* widget);

    // Font size
    void setFontScale(qreal scale); // 1.0 = normal, 1.5 = 150%, etc.
    qreal fontScale() const { return m_fontScale; }

    // Color scheme
    QPalette getPalette(ContrastMode mode) const;
    QString getStyleSheet(ContrastMode mode) const;

    // Accessibility helpers
    static bool meetsWCAGContrast(const QColor& foreground, const QColor& background);
    static qreal calculateContrastRatio(const QColor& color1, const QColor& color2);
    static qreal calculateRelativeLuminance(const QColor& color);

signals:
    void themeChanged(ContrastMode mode);
    void fontScaleChanged(qreal scale);

private:
    QPalette createHighContrastPalette() const;
    QPalette createHighContrastDarkPalette() const;
    QPalette createYellowOnBlackPalette() const;
    QPalette createWhiteOnBluePalette() const;

    QString createHighContrastStyleSheet() const;
    QString createHighContrastDarkStyleSheet() const;
    QString createYellowOnBlackStyleSheet() const;
    QString createWhiteOnBlueStyleSheet() const;

    ContrastMode m_currentMode;
    qreal m_fontScale;
};

/**
 * @brief Accessibility settings widget
 */
class AccessibilitySettings : public QWidget {
    Q_OBJECT

public:
    explicit AccessibilitySettings(QWidget* parent = nullptr);

    void setThemeManager(HighContrastTheme* themeManager);

signals:
    void settingsChanged();

private slots:
    void onContrastModeChanged(int index);
    void onFontScaleChanged(int value);
    void onEnableSoundChanged(bool checked);
    void onEnableVibrationChanged(bool checked);

private:
    void setupUI();

    HighContrastTheme* m_themeManager;
    QComboBox* m_contrastModeCombo;
    QSlider* m_fontScaleSlider;
    QCheckBox* m_enableSoundCheckbox;
    QCheckBox* m_enableVibrationCheckbox;
};

} // namespace VitalStream

#endif // HIGH_CONTRAST_THEME_H
