# Beyond the Smile — frontend image (multi-stage)
# Stage 1: build the React/Vite app
FROM node:22-alpine AS build

WORKDIR /app

# Install dependencies first (layer-cached unless package-lock.json changes)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source and build
COPY . .
RUN npm run build

# Stage 2: serve the built SPA with nginx
FROM nginx:alpine

# Copy compiled assets
COPY --from=build /app/dist /usr/share/nginx/html

# Replace the default nginx config with our custom one that:
#   - handles SPA routing (try_files -> /index.html)
#   - proxies /api/ to the backend service
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
