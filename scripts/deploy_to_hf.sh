#!/bin/bash

# Quick HF deployment script - minimal version
# Usage: ./deploy_to_hf.sh [message]

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Paths
MAIN_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_SPACE="$MAIN_PROJECT/Food101"

echo -e "${BLUE}🚀 Quick deploy to HF Spaces...${NC}"

# Copy files
echo "📁 Syncing files..."
cp -r src/ Food101/
cp streamlit/{app.py,requirements.txt,.dockerignore,config.toml} Food101/

# Note: Dockerfile is maintained separately for HF Spaces

# Git operations
cd Food101
echo "📤 Pushing to HF..."
git add .
git commit -m "${1:-Quick update}" || echo "No changes to commit"
git push

echo -e "${GREEN}✅ Deployed!${NC}"