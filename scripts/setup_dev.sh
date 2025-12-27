#!/usr/bin/env bash
set -euo pipefail

echo "== Development environment setup (macOS / Linux) =="

echo "1) Install nvm and Node 18 (if not installed)"
if ! command -v nvm >/dev/null 2>&1; then
  echo "Installing nvm..."
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
  echo "Please restart your shell or run 'source ~/.nvm/nvm.sh' and re-run this script. Exiting."
  exit 0
fi

echo "Using nvm to install Node 18"
nvm install 18
nvm use 18

echo "Installing Angular CLI globally"
npm install -g @angular/cli@latest

echo "2) Python virtual environment"
if [ ! -d "venv" ]; then
  python3 -m venv venv
  echo "Created venv/"
fi
echo "Activate with: source venv/bin/activate"

echo "Installing backend Python requirements"
if [ -f backend/requirements.txt ]; then
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r backend/requirements.txt
  deactivate
else
  echo "Warning: backend/requirements.txt not found. Create it to install backend deps."
fi

echo "Setup finished. Review docs/setup_dev.md for manual steps and VS Code extensions."
