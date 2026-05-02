#!/bin/bash
# =============================================================================
# LTX-2.3 — Script de inicio completo
# Un solo comando para levantar todo desde cero
#
# USO: bash /workspace/LTX-2/scripts/start.sh
# =============================================================================

MODELS_DIR="/workspace/models"
REPO_DIR="/workspace/LTX-2"
HF_TOKEN="${HF_TOKEN:-}"

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

header() {
    echo ""
    echo -e "${BLUE}${BOLD}========================================${NC}"
    echo -e "${BLUE}${BOLD}  $1${NC}"
    echo -e "${BLUE}${BOLD}========================================${NC}"
    echo ""
}

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
info() { echo -e "  ${YELLOW}→${NC} $1"; }
err()  { echo -e "  ${RED}✗${NC} $1"; }

# =============================================================================
# BIENVENIDA
# =============================================================================
clear
echo ""
echo -e "${BOLD}╔════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║        LTX-2.3 — Video Generator      ║${NC}"
echo -e "${BOLD}║         by elsenoralexander            ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════╝${NC}"
echo ""

# =============================================================================
# PASO 1 — Verificar que estamos en el pod correcto
# =============================================================================
header "PASO 1/4 — Verificando entorno"

if [ ! -d "/workspace" ]; then
    err "No estás dentro de un pod de RunPod."
    echo ""
    echo -e "${YELLOW}${BOLD}¿Qué tienes que hacer?${NC}"
    echo ""
    echo "  1. Ve a https://runpod.io → Pods → Deploy a Pod"
    echo "  2. Selecciona GPU: RTX 4090 (\$0.69/hr)"
    echo "  3. Template: 'RunPod Pytorch 2.4.0'"
    echo "  4. Container disk: 200 GB"
    echo "  5. Expose HTTP ports: 8888,7860"
    echo "  6. Deploy On-Demand"
    echo "  7. Cuando arranque → Connect → copia el comando SSH"
    echo "  8. Pega el comando SSH en tu terminal de Mac"
    echo "  9. Una vez dentro del pod, ejecuta:"
    echo ""
    echo -e "     ${GREEN}bash /workspace/LTX-2/scripts/start.sh${NC}"
    echo ""
    exit 1
fi

ok "Estás dentro del pod de RunPod"

# GPU
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
if [ -z "$GPU" ]; then
    err "No se detecta GPU. Algo va mal con el pod."
    exit 1
fi
ok "GPU detectada: $GPU"

# =============================================================================
# PASO 2 — Instalar dependencias (solo si no están)
# =============================================================================
header "PASO 2/4 — Verificando instalación"

DEPS_MARKER="$REPO_DIR/.deps_installed"
DEPS_ACTUAL=$(python3.12 -c "import scipy, transformers, accelerate, einops, safetensors" 2>/dev/null && echo "ok" || echo "missing")
if [ ! -f "$DEPS_MARKER" ] || [ "$DEPS_ACTUAL" = "missing" ]; then
    info "Instalando dependencias (~3 min)..."
    if [ ! -d "$REPO_DIR" ]; then
        info "Clonando repositorio..."
        git clone https://github.com/elsenoralexander/LTX-2.git "$REPO_DIR"
    fi
    cd "$REPO_DIR"
    python3.12 -m pip install -q huggingface_hub gradio
    python3.12 -m pip install -q scipy imageio imageio-ffmpeg av
    python3.12 -m pip install -q accelerate einops safetensors openimageio
    python3.12 -m pip install -q "transformers==4.51.3"
    python3.12 -m pip install -q -e packages/ltx-core --no-deps
    python3.12 -m pip install -q -e packages/ltx-pipelines --no-deps
    python3.12 -m pip install -q -e packages/ltx-trainer --no-deps
    touch "$DEPS_MARKER"
    ok "Dependencias instaladas"
else
    ok "Dependencias ya instaladas — saltando"
fi

# =============================================================================
# PASO 3 — Descargar modelos (solo si no están)
# =============================================================================
header "PASO 3/4 — Verificando modelos"

CHECKPOINT="$MODELS_DIR/ltx-2.3-22b-distilled-1.1.safetensors"
UPSCALER="$MODELS_DIR/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
LORA="$MODELS_DIR/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
GEMMA="$MODELS_DIR/gemma-3-12b"

MODELS_OK=true

[ ! -f "$CHECKPOINT" ] && MODELS_OK=false
[ ! -f "$UPSCALER" ]   && MODELS_OK=false
[ ! -f "$LORA" ]       && MODELS_OK=false
[ ! -d "$GEMMA" ]      && MODELS_OK=false

if [ "$MODELS_OK" = false ]; then
    info "Modelos no encontrados — descargando (~80 GB, 15-20 min)..."
    echo ""

    # Verificar HF_TOKEN
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${YELLOW}${BOLD}Necesitas tu token de HuggingFace.${NC}"
        echo ""
        echo "  1. Ve a https://huggingface.co/settings/tokens"
        echo "  2. Crea un token tipo 'Read'"
        echo "  3. Pégalo aquí:"
        echo ""
        read -r -p "  HF_TOKEN: " HF_TOKEN
        export HF_TOKEN
    fi

    pip install -q huggingface_hub hf_transfer
    export HF_HUB_ENABLE_HF_TRANSFER=1

    # Login con token
    python3 -c "from huggingface_hub import login; login('$HF_TOKEN')" > /dev/null 2>&1

    mkdir -p "$MODELS_DIR"

    info "Descargando modelo distilled (~44 GB)..."
    python3 -c "
from huggingface_hub import hf_hub_download
import sys, os
print('  Iniciando descarga...', flush=True)
path = hf_hub_download('Lightricks/LTX-2.3', 'ltx-2.3-22b-distilled-1.1.safetensors', local_dir='$MODELS_DIR')
size = os.path.getsize(path) / 1024**3
print(f'  Descargado: {size:.1f} GB -> {path}', flush=True)
"

    info "Descargando spatial upscaler (~2 GB)..."
    python3 -c "
from huggingface_hub import hf_hub_download
import sys, os
print('  Iniciando descarga...', flush=True)
path = hf_hub_download('Lightricks/LTX-2.3', 'ltx-2.3-spatial-upscaler-x2-1.1.safetensors', local_dir='$MODELS_DIR')
size = os.path.getsize(path) / 1024**3
print(f'  Descargado: {size:.2f} GB -> {path}', flush=True)
"

    info "Descargando distilled LoRA (~1 GB)..."
    python3 -c "
from huggingface_hub import hf_hub_download
import sys, os
print('  Iniciando descarga...', flush=True)
path = hf_hub_download('Lightricks/LTX-2.3', 'ltx-2.3-22b-distilled-lora-384-1.1.safetensors', local_dir='$MODELS_DIR')
size = os.path.getsize(path) / 1024**3
print(f'  Descargado: {size:.2f} GB -> {path}', flush=True)
"

    info "Descargando Gemma text encoder (~8 GB)..."
    python3 -c "
from huggingface_hub import snapshot_download
import os
print('  Iniciando descarga (varios ficheros)...', flush=True)
path = snapshot_download('google/gemma-3-12b-it-qat-q4_0-unquantized', local_dir='$GEMMA')
total = sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(path) for f in fs) / 1024**3
print(f'  Descargado: {total:.1f} GB -> {path}', flush=True)
"

    ok "Modelos descargados"
else
    ok "Modelos ya descargados — saltando"
    ok "  Checkpoint:  $CHECKPOINT"
    ok "  Upscaler:    $UPSCALER"
    ok "  LoRA:        $LORA"
    ok "  Gemma:       $GEMMA"
fi

# =============================================================================
# PASO 4 — Lanzar Web UI
# =============================================================================
header "PASO 4/4 — Lanzando Web UI"

cd "$REPO_DIR"

# Obtener la URL pública del pod
POD_ID=$(hostname | cut -d'-' -f1 2>/dev/null || echo "tu-pod")
echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║   Web UI lista! Ábrela en tu navegador:      ║"
echo "  ║                                              ║"
echo "  ║   → Ve a RunPod → tu pod → Connect          ║"
echo "  ║   → Pulsa el enlace del puerto 7860          ║"
echo "  ║                                              ║"
echo "  ║   (Ctrl+C para parar)                        ║"
echo "  ╚══════════════════════════════════════════════╝"
echo -e "${NC}"

MODELS_DIR="$MODELS_DIR" REPO_DIR="$REPO_DIR" python3.12 "$REPO_DIR/scripts/webui.py"
