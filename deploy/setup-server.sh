#!/usr/bin/env bash
# Run this ONCE on a fresh Hetzner VPS (Ubuntu 22.04/24.04)
# Usage: ssh root@YOUR_SERVER_IP < deploy/setup-server.sh
set -euo pipefail

echo "=== Installing Docker ==="
apt-get update
apt-get install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

echo "=== Creating deploy user ==="
useradd -m -s /bin/bash -G docker deploy || true
mkdir -p /home/deploy/.ssh
cp /root/.ssh/authorized_keys /home/deploy/.ssh/ 2>/dev/null || true
chown -R deploy:deploy /home/deploy/.ssh
chmod 700 /home/deploy/.ssh
chmod 600 /home/deploy/.ssh/authorized_keys 2>/dev/null || true

echo "=== Creating app directory ==="
mkdir -p /opt/polymarket-bot
chown deploy:deploy /opt/polymarket-bot

echo "=== Enabling Docker on boot ==="
systemctl enable docker
systemctl start docker

echo ""
echo "=== Done! ==="
echo "Next steps:"
echo "  1. Generate a deploy SSH key:  ssh-keygen -t ed25519 -f deploy_key -N ''"
echo "  2. Add the public key to server:  ssh-copy-id -i deploy_key.pub deploy@YOUR_SERVER_IP"
echo "  3. Add these GitHub repo secrets:"
echo "     - DEPLOY_HOST: your server IP"
echo "     - DEPLOY_SSH_KEY: contents of deploy_key (private key)"
echo "     - DEPLOY_ENV: contents of your production .env file"
