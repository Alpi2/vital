#include <QApplication>
#include <QScreen>
#include <QStyleFactory>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    // Enable high DPI scaling
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
    
    QApplication app(argc, argv);
    
    // Set application metadata
    app.setApplicationName("VitalStream Central Station");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("VitalStream Medical Systems");
    app.setOrganizationDomain("vitalstream.com");
    
    // Set modern style
    app.setStyle(QStyleFactory::create("Fusion"));
    
    // Create and show main window
    MainWindow mainWindow;
    
    // Check for multi-monitor setup
    QList<QScreen*> screens = QApplication::screens();
    if (screens.size() > 1) {
        // Position on primary screen
        mainWindow.setGeometry(screens[0]->geometry());
    } else {
        mainWindow.showMaximized();
    }
    
    mainWindow.show();
    
    return app.exec();
}
