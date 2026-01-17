FROM ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libdcmtk-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    libgrpc++-dev \
    libgrpc-dev \
    libspdlog-dev \
    libssl-dev \
    uuid-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source code, proto files, and wrapper scripts
COPY ./dicom-service .
COPY ./protos ./protos
COPY ./protoc_wrapper.sh .
COPY ./grpc_wrapper.sh .

# Build
RUN rm -rf build && mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --parallel $(nproc)

# Runtime stage
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libdcmtk16 \
    libprotobuf23 \
    libgrpc++ \
    libspdlog1 \
    libssl3 \
    uuid-runtime \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /build/build/dicom_service /usr/local/bin/

# Create storage directory
RUN mkdir -p /data/dicom/studies /data/dicom/temp /data/dicom/trash /data/dicom/logs && \
    chmod -R 755 /data/dicom

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD echo "Health check" || exit 1

# Run service
CMD ["/usr/local/bin/dicom_service", "--address", "0.0.0.0:50051", "--storage", "/data/dicom"]
