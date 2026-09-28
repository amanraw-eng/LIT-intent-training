#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
TMPL_DIR="$REPO_DIR/service_files"

# Service user = the owner of the repo dir on disk. Robust under sudo
# (whoami would return root) and across machines, without needing a vendor arg.
SERVICE_USER="$(stat -c '%U' "$REPO_DIR")"

UNITS=(litintent-api)

for u in "${UNITS[@]}"; do
  tmpl="$TMPL_DIR/${u}.service.tmpl"
  if [[ ! -f "$tmpl" ]]; then
    echo "Error: $tmpl not found" >&2
    exit 1
  fi
done

if [[ ! -f "$REPO_DIR/.env" ]]; then
  echo "Error: $REPO_DIR/.env not found." >&2
  echo "        Copy src/.env.example to $REPO_DIR/.env and set at least HF_TOKEN" >&2
  echo "        (needed if HF_MODEL_REPO is a private Hugging Face repo)." >&2
  exit 1
fi

echo "Checking system dependencies:"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "  ffmpeg not found — installing via apt (used by Whisper to decode uploaded audio)"
  sudo apt-get update -qq
  sudo apt-get install -y ffmpeg
else
  echo "  ffmpeg: $(command -v ffmpeg)"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "  uv not found — installing"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "  uv: $(command -v uv) ($(uv --version))"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
  echo "  Warning: nvidia-smi not found — DEVICE=cuda in constants.py will fall back to CPU." >&2
fi
echo

# pyproject.toml / uv.lock live at the repo root, so the venv (.venv) is
# created there too, with app/ (in src/) installed into it editable — that's
# why ExecStart below can run "python -m app.main" from src/ with no extra
# PYTHONPATH wiring.
echo "Installing Python dependencies (uv sync):"
(cd "$REPO_DIR" && uv sync)
echo

echo "Rendering unit files:"
echo "  User=$SERVICE_USER"
echo "  WorkingDirectory=$REPO_DIR/src"
echo

for u in "${UNITS[@]}"; do
  rendered=$(mktemp)
  sed \
    -e "s|\${SERVICE_USER}|$SERVICE_USER|g" \
    -e "s|\${REPO_DIR}|$REPO_DIR|g" \
    "$TMPL_DIR/${u}.service.tmpl" > "$rendered"
  sudo install -m 0644 "$rendered" "/etc/systemd/system/${u}.service"
  rm -f "$rendered"
  echo "  /etc/systemd/system/${u}.service"
done

sudo systemctl daemon-reload
sudo systemctl enable "${UNITS[@]/%/.service}"

echo
echo "Next:"
echo "  sudo systemctl start litintent-api"
echo "  sudo journalctl -u litintent-api -f"
echo "  curl http://localhost:8000/health"
echo
echo "(Re-run install.sh after pulling changes that touch service_files/*.tmpl,"
echo " or after changing dependencies in pyproject.toml.)"
