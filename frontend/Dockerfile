# frontend/Dockerfile
# Build stage
FROM node:18-alpine AS build

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build Angular app
RUN npm run build -- --configuration production

# Production stage
FROM nginx:alpine

# Copy built app
COPY --from=build /app/dist/vitalstream-frontend /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Copy wasm files
COPY --from=build /app/src/assets/wasm /usr/share/nginx/html/assets/wasm

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
