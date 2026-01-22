#!/bin/bash

BUILD_DIR=$1
PROTO_NAME=$2

cd /Users/alper/Desktop/GitHubProjeleri/vital

# Generate gRPC files correctly
if [ "$PROTO_NAME" = "dicom_service" ]; then
    protoc --grpc_out=${BUILD_DIR} --plugin=/opt/homebrew/bin/protoc-gen-grpc -Iprotos protos/dicom/v1/dicom_service.proto
    protoc --grpc_out=${BUILD_DIR} --plugin=/opt/homebrew/bin/protoc-gen-grpc -Iprotos protos/common/v1/common.proto
    
    # Move files to correct location
    if [ -f "${BUILD_DIR}/dicom/v1/dicom_service.grpc.pb.h" ]; then
        cp "${BUILD_DIR}/dicom/v1/dicom_service.grpc.pb.h" "${BUILD_DIR}/dicom_service.grpc.pb.h"
        cp "${BUILD_DIR}/dicom/v1/dicom_service.grpc.pb.cc" "${BUILD_DIR}/dicom_service.grpc.pb.cc"
    fi
    
    if [ -f "${BUILD_DIR}/common/v1/common.grpc.pb.h" ]; then
        cp "${BUILD_DIR}/common/v1/common.grpc.pb.h" "${BUILD_DIR}/common.grpc.pb.h"
        cp "${BUILD_DIR}/common/v1/common.grpc.pb.cc" "${BUILD_DIR}/common.grpc.pb.cc"
    fi
fi
