#!/bin/bash
set -e

# Install Docker
apt-get update
apt-get install -y docker.io awscli

# Start Docker service
systemctl start docker
systemctl enable docker

# Mount EBS volume for models
mkfs -t ext4 /dev/xvdf
mkdir -p /mnt/models
mount /dev/xvdf /mnt/models

# Add mount to fstab
echo "/dev/xvdf /mnt/models ext4 defaults,nofail 0 2" >> /etc/fstab

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Configure Ollama model path
echo 'OLLAMA_MODELS="/mnt/models"' > /etc/ollama/env

# Login to ECR
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${ecr_repository}

# Pull and run application
docker pull ${ecr_repository}:${image_tag}
docker run -d \
  --name rag-app \
  --restart always \
  -v /mnt/models:/mnt/models \
  ${ecr_repository}:${image_tag}