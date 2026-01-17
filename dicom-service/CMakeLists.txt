cmake_minimum_required(VERSION 3.20)
project(VitalStreamDICOMService VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard and Configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Building VitalStream DICOM Service in ${CMAKE_BUILD_TYPE} mode")

# Compiler options
set(COMMON_FLAGS "-Wall -Wextra -Wpedantic")
set(DEBUG_FLAGS "-g -O0")
set(RELEASE_FLAGS "-O3 -DNDEBUG -march=native")

# Find required packages using pkg-config
find_package(PkgConfig REQUIRED)

# Find packages
pkg_check_modules(PROTOBUF REQUIRED protobuf)
pkg_check_modules(GRPC REQUIRED grpc++)
find_package(DCMTK REQUIRED COMPONENTS ofstd oflog oficonv dcmdata dcmnet dcmimgle dcmimage)
find_package(Threads REQUIRED)

# Find other packages
find_package(spdlog REQUIRED)
find_package(OpenSSL REQUIRED)
pkg_check_modules(UUID REQUIRED uuid)

# Package status reporting
if(DCMTK_FOUND)
    message(STATUS "✅ DCMTK found: ${DCMTK_VERSION}")
else()
    message(FATAL_ERROR "❌ DCMTK not found - required for DICOM processing")
endif()

message(STATUS "✅ Protobuf: ${PROTOBUF_VERSION}")
message(STATUS "✅ gRPC: ${GRPC_VERSION}")

# Proto files configuration
set(PROTO_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../protos/dicom/v1/dicom_service.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/../protos/common/v1/common.proto
)

# Generate protobuf and gRPC code
message(STATUS "🔧 Generating protobuf and gRPC code...")

set(PROTO_GENERATED_SOURCES)
set(PROTO_GENERATED_HEADERS)

foreach(proto_file ${PROTO_FILES})
    get_filename_component(proto_name ${proto_file} NAME_WE)
    
    # Generate protobuf
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.pb.h
               ${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.pb.cc
        COMMAND protoc --cpp_out=${CMAKE_CURRENT_BINARY_DIR} --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/../protos ${proto_file}
        DEPENDS ${proto_file}
        COMMENT "Generating protobuf for ${proto_name}"
    )
    
    list(APPEND PROTO_GENERATED_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.pb.h)
    list(APPEND PROTO_GENERATED_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.pb.cc)
    
    # Generate gRPC
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.grpc.pb.h
               ${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.grpc.pb.cc
        COMMAND protoc --grpc_out=${CMAKE_CURRENT_BINARY_DIR} --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/../protos ${proto_file}
        DEPENDS ${proto_file}
        COMMENT "Generating gRPC for ${proto_name}"
    )
    
    list(APPEND PROTO_GENERATED_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.grpc.pb.h)
    list(APPEND PROTO_GENERATED_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.grpc.pb.cc)
endforeach()

# Set generated files as source variables
set(PROTO_SRCS ${PROTO_GENERATED_SOURCES})
set(GRPC_SRCS ${PROTO_GENERATED_SOURCES})

# Main executable
add_executable(dicom_service
    src/main.cpp
    src/dicom_service_impl.cpp
    src/storage_manager.cpp
    src/audit_logger.cpp
    # src/waveform_extractor.cpp  # Moved to dicom_service_impl.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/dicom/v1/dicom_service.pb.cc
    ${CMAKE_CURRENT_BINARY_DIR}/dicom/v1/dicom_service.grpc.pb.cc
    ${CMAKE_CURRENT_BINARY_DIR}/common/v1/common.pb.cc
    ${CMAKE_CURRENT_BINARY_DIR}/common/v1/common.grpc.pb.cc
)

# Header files
set(HEADERS
    include/dicom_service_impl.h
    include/waveform_extractor.h
    include/storage_manager.h
    include/audit_logger.h
    # include/health_service.h  # Disabled for now
)

# Target-specific include directories
target_include_directories(dicom_service
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
        ${DCMTK_INCLUDE_DIRS}
        ${PROTOBUF_INCLUDE_DIRS}
        ${GRPC_INCLUDE_DIRS}
)

# Compiler-specific options
target_compile_options(dicom_service
    PRIVATE
        -Wall -Wextra -Wpedantic -Wno-nullability-extension
        -Wno-gcc-compat
        -Wno-deprecated-declarations
        -Wno-unused-parameter
        $<$<CONFIG:Debug>:-g -O0>
        $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>
)

# Link libraries
target_link_libraries(dicom_service
    PRIVATE
        # DICOM processing - check if imported targets are available
        $<$<TARGET_EXISTS:DCMTK::ofstd>:DCMTK::ofstd>
        $<$<TARGET_EXISTS:DCMTK::oflog>:DCMTK::oflog>
        $<$<TARGET_EXISTS:DCMTK::oficonv>:DCMTK::oficonv>
        $<$<TARGET_EXISTS:DCMTK::dcmdata>:DCMTK::dcmdata>
        $<$<TARGET_EXISTS:DCMTK::dcmnet>:DCMTK::dcmnet>
        $<$<TARGET_EXISTS:DCMTK::dcmimgle>:DCMTK::dcmimgle>
        $<$<TARGET_EXISTS:DCMTK::dcmimage>:DCMTK::dcmimage>
        $<$<NOT:$<TARGET_EXISTS:DCMTK::ofstd>>:${DCMTK_LIBRARIES}>
        
        # Protobuf and gRPC
        ${PROTOBUF_LIBRARIES}
        ${GRPC_LIBRARIES}
        
        # System libraries
        Threads::Threads
        spdlog::spdlog
        
        # Security
        OpenSSL::SSL
        OpenSSL::Crypto
        
        # Utilities
        ${CMAKE_DL_LIBS}
        ${UUID_LIBRARIES}
)

# Add library directories for proper linking
target_link_directories(dicom_service PRIVATE 
    ${PROTOBUF_LIBRARY_DIRS}
    ${GRPC_LIBRARY_DIRS}
    ${UUID_LIBRARY_DIRS}
)

# Platform-specific libraries
if(UNIX AND NOT APPLE)
    target_link_libraries(dicom_service PRIVATE rt)
endif()

# Installation configuration
include(GNUInstallDirs)

install(TARGETS dicom_service
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vitalstream/dicom
)

# Testing configuration
enable_testing()

# Add Google Test if available
find_package(GTest QUIET)
if(GTest_FOUND)
    message(STATUS "✅ Google Test found - enabling unit tests")
    
    # Simple test executable
    add_executable(simple_test
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/simple_test.cpp
    )
    
    target_link_libraries(simple_test
        PRIVATE
            GTest::gtest
            GTest::gtest_main
    )
    
    # Test executables
    add_executable(dicom_service_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dicom_service.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/performance_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/dicom_validation_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dicom_service_impl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/storage_manager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audit_logger.cpp
        # src/waveform_extractor.cpp  # Moved to dicom_service_impl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/dicom/v1/dicom_service.pb.cc
        ${CMAKE_CURRENT_BINARY_DIR}/dicom/v1/dicom_service.grpc.pb.cc
        ${CMAKE_CURRENT_BINARY_DIR}/common/v1/common.pb.cc
        ${CMAKE_CURRENT_BINARY_DIR}/common/v1/common.grpc.pb.cc
    )
    
    # Include directories for tests
    target_include_directories(dicom_service_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}
            ${DCMTK_INCLUDE_DIRS}
            ${PROTOBUF_INCLUDE_DIRS}
            ${GRPC_INCLUDE_DIRS}
    )
    
    # Add library directories for proper linking
    target_link_directories(dicom_service_tests PRIVATE 
        ${PROTOBUF_LIBRARY_DIRS}
        ${GRPC_LIBRARY_DIRS}
        ${UUID_LIBRARY_DIRS}
    )
    
    target_link_libraries(dicom_service_tests
        PRIVATE
            GTest::gtest
            GTest::gtest_main
            # DICOM processing - use imported targets
            $<$<TARGET_EXISTS:DCMTK::ofstd>:DCMTK::ofstd>
            $<$<TARGET_EXISTS:DCMTK::oflog>:DCMTK::oflog>
            $<$<TARGET_EXISTS:DCMTK::oficonv>:DCMTK::oficonv>
            $<$<TARGET_EXISTS:DCMTK::dcmdata>:DCMTK::dcmdata>
            $<$<TARGET_EXISTS:DCMTK::dcmnet>:DCMTK::dcmnet>
            $<$<TARGET_EXISTS:DCMTK::dcmimgle>:DCMTK::dcmimgle>
            $<$<TARGET_EXISTS:DCMTK::dcmimage>:DCMTK::dcmimage>
            $<$<NOT:$<TARGET_EXISTS:DCMTK::ofstd>>:${DCMTK_LIBRARIES}>
            
            # Protobuf and gRPC
            ${PROTOBUF_LIBRARIES}
            ${GRPC_LIBRARIES}
            
            # System libraries
            spdlog::spdlog
            Threads::Threads
        )  
    # Add test to CTest
    add_test(NAME DICOMServiceUnitTests COMMAND dicom_service_tests)
    
    # Set test properties
    set_tests_properties(DICOMServiceUnitTests PROPERTIES
        TIMEOUT 300  # 5 minutes timeout for performance tests
    )
    
    message(STATUS "✅ Unit tests configured")
else()
    message(STATUS "⚠️ Google Test not found - unit tests disabled")
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "🏥 VitalStream DICOM Service Configuration Summary:")
message(STATUS "   Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "   C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "   Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "   Source Directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "   Binary Directory: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "")
message(STATUS "📦 Dependencies:")
message(STATUS "   DCMTK: ${DCMTK_VERSION}")
message(STATUS "   Protobuf: ${PROTOBUF_VERSION}")
message(STATUS "   gRPC: ${GRPC_VERSION}")
message(STATUS "   spdlog: ${spdlog_VERSION}")
message(STATUS "")
message(STATUS "🔧 Features:")
message(STATUS "   Unit Tests: ${GTEST_FOUND}")
message(STATUS "")
