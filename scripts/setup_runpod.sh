#!/bin/bash
# =============================================================================
# LTX-2.3 — Setup Script for RunPod (CUDA 12.7+, Python 3.12)
# Run this ONCE after launching your pod:  bash scripts/setup_runpod.sh
# =============================================================================
set -e

REPO_DIR="/workspace/LTX-2"
MODELS_DIR="/workspace/models"

echo "=== [1/5] Actualizando sistema ==="
apt-get update -qq && apt-get install -y -qq git curl wget ffmpeg

echo "=== [2/5] Instalando uv (gestor de paquetes) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== [3/5] Clonando repositorio LTX-2 ==="
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/Lightricks/LTX-2.git "$REPO_DIR"
else
    echo "  -> Repositorio ya existe, haciendo pull..."
    cd "$REPO_DIR" && git pull
fi

cd "$REPO_DIR"

echo "=== [4/5] Instalando dependencias Python ==="
uv sync --frozen
# Flash Attention para Hopper GPUs (H100). Omitir si usas A100/RTX:
# uv sync --extra xformers

echo "=== [5/5] Creando directorio de modelos ==="
mkdir -p "$MODELS_DIR"

echo ""
echo "=========================================="
echo " Setup completado!"
echo " Siguiente paso: bash scripts/download_models.sh"
echo "=========================================="
