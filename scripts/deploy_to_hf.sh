#!/bin/bash

# Quick HF deployment script - minimal version
# Usage: ./deploy_to_hf.sh [message]

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Resolve repo root (works whether run from repo root or elsewhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HF_SPACE="$REPO_ROOT/Food101-Streamlit"

echo -e "${BLUE}🚀 Quick deploy to HF Spaces...${NC}"

# Copy files
echo "📁 Syncing files..."
cp "$REPO_ROOT/pyproject.toml" "$HF_SPACE/"
cp "$REPO_ROOT/uv.lock" "$HF_SPACE/"
rsync -a --delete "$REPO_ROOT/src/" "$HF_SPACE/src/"

# Git operations
pushd "$HF_SPACE" >/dev/null
echo "📤 Pushing to HF..."
	# Proactively remove any previously committed large model files from the Space repo
	if [ -d "models" ]; then
		echo "🧹 Removing previously committed models/ from Space repo to avoid LFS overages"
		rm -rf models
	fi
git add .
if git commit -m "${1:-Quick update}"; then
	:
else
	echo "No changes to commit; creating empty commit to trigger rebuild"
	git commit --allow-empty -m "${1:-Trigger rebuild}"
fi
git push
popd >/dev/null

# Clean up copied src folder to avoid confusion locally and save disk space
rm -rf "$HF_SPACE/src/"

echo -e "${GREEN}✅ Deployed!${NC}"