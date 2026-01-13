# VitalStream Central Station

## Overview

Central Station is a high-performance desktop application for monitoring multiple patients simultaneously in clinical settings.

## Features

- **Multi-Patient Monitoring:** Monitor 16+ patients simultaneously
- **Real-time Waveforms:** 6-channel ECG waveform rendering at 60 FPS
- **Alarm Management:** Multi-level alarm system with intelligent filtering
- **Multi-Display Support:** Optimized for 2-4 monitor setups
- **12-Lead ECG:** Standard and Cabrera format display
- **Trend Analysis:** 1-72 hour trend visualization

## Technology Stack

- **Framework:** Qt 6 (C++17)
- **Rendering:** Qt Quick / QML + OpenGL
- **Charts:** QCustomPlot
- **Audio:** Qt Multimedia
- **Network:** Qt WebSockets
- **Build System:** CMake

## Requirements

### System Requirements
- **OS:** Windows 10/11, Linux (Ubuntu 20.04+), macOS 11+
- **CPU:** Intel i5 or equivalent (quad-core recommended)
- **RAM:** 8GB minimum, 16GB recommended
- **GPU:** OpenGL 3.3+ support
- **Display:** 1920x1080 minimum, multi-monitor recommended

### Development Requirements
- Qt 6.5+
- CMake 3.16+
- C++17 compatible compiler (GCC 9+, Clang 10+, MSVC 2019+)
- Qt Creator (recommended IDE)

## Building

### Linux/macOS

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Windows

```bash
mkdir build
cd build
cmake -G "Visual Studio 16 2019" ..
cmake --build . --config Release
```

## Architecture

### Core Components

1. **MainWindow**
   - Main application window
   - Multi-display management
   - Layout management

2. **PatientGridView**
   - Grid layout for multiple patients
   - Patient prioritization
   - Quick view dashboard

3. **WaveformRenderer**
   - High-performance waveform rendering
   - OpenGL acceleration
   - 60 FPS target
   - Zoom, pan, freeze functionality

4. **AlarmManager**
   - Multi-level alarm system
   - Alarm acknowledgment
   - Escalation logic
   - Audio/visual alerts

5. **WebSocketClient**
   - Real-time data communication
   - Connection management
   - Automatic reconnection

6. **PatientData**
   - Patient information model
   - Vital signs data
   - Alarm state

### Data Flow

```
Backend (Rust) 
    ↓ WebSocket
WebSocketClient
    ↓
PatientData (Model)
    ↓
PatientGridView + WaveformRenderer
    ↓
Display (Qt Widgets/QML)
```

## Performance Targets

- **Waveform Rendering:** 60 FPS for 96 channels (16 patients x 6 channels)
- **Latency:** <100ms from data reception to display
- **Memory Usage:** <500MB for 16 patients
- **CPU Usage:** <20% on recommended hardware

## Configuration

Configuration file: `config.json`

```json
{
  "backend": {
    "url": "ws://localhost:8080",
    "reconnect_interval": 5000
  },
  "display": {
    "max_patients": 16,
    "waveform_channels": 6,
    "refresh_rate": 60
  },
  "alarms": {
    "audio_enabled": true,
    "volume": 80,
    "escalation_timeout": 300
  }
}
```

## Development Status

- [x] Project structure created
- [x] CMake configuration
- [ ] Main window implementation
- [ ] Patient grid view
- [ ] Waveform renderer
- [ ] Alarm manager
- [ ] WebSocket client
- [ ] Multi-display support
- [ ] 12-lead ECG display
- [ ] Trend visualization

## License

Proprietary - VitalStream Medical Systems

## Contact

For questions or support, contact: dev@vitalstream.com
