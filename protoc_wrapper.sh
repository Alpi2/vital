#!/bin/bash

BUILD_DIR=$1
PROTO_NAME=$2

cd /Users/alper/Desktop/GitHubProjeleri/vital

# Generate protobuf files correctly
if [ "$PROTO_NAME" = "dicom_service" ]; then
    protoc --cpp_out=${BUILD_DIR} -Iprotos protos/dicom/v1/dicom_service.proto
    protoc --cpp_out=${BUILD_DIR} -Iprotos protos/common/v1/common.proto
    
    # Move files to correct location
    if [ -f "${BUILD_DIR}/dicom/v1/dicom_service.pb.h" ]; then
        cp "${BUILD_DIR}/dicom/v1/dicom_service.pb.h" "${BUILD_DIR}/dicom_service.pb.h"
        cp "${BUILD_DIR}/dicom/v1/dicom_service.pb.cc" "${BUILD_DIR}/dicom_service.pb.cc"
    fi
    
    if [ -f "${BUILD_DIR}/common/v1/common.pb.h" ]; then
        cp "${BUILD_DIR}/common/v1/common.pb.h" "${BUILD_DIR}/common.pb.h"
        cp "${BUILD_DIR}/common/v1/common.pb.cc" "${BUILD_DIR}/common.pb.cc"
    fi
fi
