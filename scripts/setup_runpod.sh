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
# Usamos pip en lugar de uv para reutilizar PyTorch del sistema (ya viene en el template)
# Esto evita descargar 3-4 GB de CUDA que ya están instalados
pip install -q huggingface_hub gradio
pip install -q -e packages/ltx-core --no-deps
pip install -q -e packages/ltx-pipelines --no-deps
pip install -q -e packages/ltx-trainer --no-deps
# Instalar solo las dependencias que NO son torch/CUDA
pip install -q wandb imageio-ffmpeg scipy bitsandbytes opencv-python av 2>/dev/null || true

echo "=== [5/5] Creando directorio de modelos ==="
mkdir -p "$MODELS_DIR"

echo ""
echo "=========================================="
echo " Setup completado!"
echo " Siguiente paso: bash scripts/download_models.sh"
echo "=========================================="
