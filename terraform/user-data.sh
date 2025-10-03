#!/bin/bash
set -e

region="${region}"
ecr_registry="${ecr_registry}"
ecr_repository="${ecr_repository}"
image_tag="${image_tag}"

# --- Install Docker & AWS CLI ---
apt-get update -y
apt-get install -y docker.io awscli curl
systemctl enable --now docker

# --- Mount EBS volume for models ---
if ! blkid /dev/xvdf; then
  mkfs -t ext4 /dev/xvdf
fi
mkdir -p /mnt/models
mount /dev/xvdf /mnt/models
grep -q "/mnt/models" /etc/fstab || echo "/dev/xvdf /mnt/models ext4 defaults,nofail 0 2" >> /etc/fstab

# --- Install Ollama ---
curl -fsSL https://ollama.com/install.sh | sh

# Configure Ollama model path
mkdir -p /etc/systemd/system/ollama.service.d
cat <<EOF > /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_MODELS=/mnt/models"
EOF
systemctl daemon-reexec
systemctl restart ollama

# --- Login to ECR ---
aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "$ecr_registry"

# --- Pull and run application ---
docker pull "$ecr_registry/$ecr_repository:$image_tag"
docker rm -f rag-app || true
docker run -d \
  --name rag-app \
  --restart always \
  -p 8000:8000 \
  -v /mnt/models:/mnt/models \
  "$ecr_registry/$ecr_repository:$image_tag"
