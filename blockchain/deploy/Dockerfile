# Multi-stage build for VitalStream Blockchain Node
FROM rust:1.75-slim as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    cmake \
    git \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy source code
COPY blockchain/ ./blockchain/
COPY Cargo.toml Cargo.lock ./

# Build release binary
RUN cargo build --release --package blockchain

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u 1000 -U -s /bin/sh -d /vitalstream vitalstream

# Copy binary from builder
COPY --from=builder /build/target/release/vitalstream-node /usr/local/bin/

# Create directories
RUN mkdir -p /var/lib/vitalstream /etc/vitalstream /var/log/vitalstream && \
    chown -R vitalstream:vitalstream /var/lib/vitalstream /var/log/vitalstream

# Switch to non-root user
USER vitalstream

# Expose ports
EXPOSE 30333 9933 9944 9615

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9933/health || exit 1

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/vitalstream-node"]
