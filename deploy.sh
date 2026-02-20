#!/bin/bash
# Deployment script for Mimic Master

set -e

# Configuration (can be overridden via environment)
SERVER_HOST="${SERVER_HOST:-your-server.com}"
SERVER_USER="${SERVER_USER:-root}"
SERVER_PATH="${SERVER_PATH:-/opt/mimic-master}"

echo "Deploying to $SERVER_USER@$SERVER_HOST:$SERVER_PATH..."

# Build Docker image
echo "Building Docker image..."
docker build -t mimic-master:latest .

# Save image to tar
echo "Exporting Docker image..."
docker save mimic-master:latest | gzip > mimic-master.tar

# Transfer to server
echo "Transferring to server..."
scp mimic-master.tar $SERVER_USER@$SERVER_HOST:/tmp/

# Clean up local tar
rm mimic-master.tar

# Deploy on server
echo "Deploying on server..."
ssh $SERVER_USER@$SERVER_HOST << 'ENDSSH'
  # Load Docker image
  docker load < /tmp/mimic-master.tar
  rm /tmp/mimic-master.tar

  # Stop and remove old container
  docker stop mimic-master || true
  docker rm mimic-master || true

  # Run new container
  docker run -d \
    --name mimic-master \
    --restart unless-stopped \
    -p 8000:8000 \
    --env-file /opt/mimic-master/.env \
    mimic-master:latest
ENDSSH

echo "Deployment complete!"
echo "Service should be available at http://$SERVER_HOST:8000/api/v1/health"
