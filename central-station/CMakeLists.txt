cmake_minimum_required(VERSION 3.16)

project(VitalStreamCentralStation VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Quick Multimedia Network WebSockets Charts)

# Source files
set(SOURCES
    src/main.cpp
    src/mainwindow.cpp
    src/patientgridview.cpp
    src/waveformrenderer.cpp
    src/alarmmanager.cpp
    src/websocketclient.cpp
    src/patientdata.cpp
)

set(HEADERS
    include/mainwindow.h
    include/patientgridview.h
    include/waveformrenderer.h
    include/alarmmanager.h
    include/websocketclient.h
    include/patientdata.h
)

set(UI_FILES
    ui/mainwindow.ui
)

set(RESOURCES
    resources/resources.qrc
)

add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${HEADERS}
    ${UI_FILES}
    ${RESOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Quick
    Qt6::Multimedia
    Qt6::Network
    Qt6::WebSockets
    Qt6::Charts
)

# Enable warnings
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Installation
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)
