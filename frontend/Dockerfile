# frontend/Dockerfile
# Build stage
FROM node:20-alpine AS build

WORKDIR /app

# Copy package files
COPY package*.json ./
# Remove package-lock to allow install to fetch platform-appropriate native binaries
RUN rm -f package-lock.json || true
# Ensure npm installs platform-specific binaries for the container (linux/arm64)
ENV npm_config_platform=linux
ENV npm_config_arch=arm64
RUN npm install --only=production
# Ensure the Linux esbuild binary is present (handles cases where lockfile references darwin binary)
RUN npm install @esbuild/linux-arm64 --no-save || true
# Rebuild esbuild for the container platform in case package-lock was generated on macOS
RUN npm rebuild esbuild
# Unset the environment variables to avoid leaking into later stages
ENV npm_config_platform=
ENV npm_config_arch=

# Copy source code
COPY . .

# Build Angular app
RUN npm run build -- --configuration production
# Ensure assets/wasm directory exists even when no wasm emitted to avoid COPY failure
RUN mkdir -p dist/vitalstream-frontend/browser/assets/wasm

# Production stage
FROM nginx:alpine

# Copy built app (use browser/ output for Angular build)
COPY --from=build /app/dist/vitalstream-frontend/browser /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Copy wasm files from build output
COPY --from=build /app/dist/vitalstream-frontend/browser/assets/wasm /usr/share/nginx/html/assets/wasm

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
