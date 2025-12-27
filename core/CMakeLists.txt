cmake_minimum_required(VERSION 3.13)
project(vital_core)

add_library(vital_core STATIC src/core.cpp src/ecg_generator.cpp src/ecg_analyzer.cpp src/bindings.cpp)
target_include_directories(vital_core PUBLIC include)

set_target_properties(vital_core PROPERTIES
    CXX_STANDARD 17
)

# For WebAssembly builds, toolchain and flags should be configured separately.
