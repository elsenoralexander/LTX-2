#!/bin/bash
# =============================================================================
# LTX-2.3 — Descarga de modelos desde HuggingFace
# Requiere: pip install huggingface_hub  +  huggingface-cli login
#
# Uso:  bash scripts/download_models.sh [--fast | --full | --minimal]
#   --minimal  Solo distilled (más rápido, menos VRAM, calidad OK)
#   --fast     Distilled + upscaler x2   (recomendado para empezar)
#   --full     Todo (dev + distilled + upscalers + LoRAs)  [default]
# =============================================================================
set -e

MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODE="${1:---full}"

mkdir -p "$MODELS_DIR"

echo "=== Modo: $MODE | Destino: $MODELS_DIR ==="

# Asegurar que huggingface_hub esté instalado
pip install -q huggingface_hub

# --- Función helper ---
hf_download() {
    local repo="$1"
    local file="$2"
    local dest="$3"
    echo "  Descargando $file..."
    huggingface-cli download "$repo" "$file" --local-dir "$dest" --local-dir-use-symlinks False
}

# =============================================================================
# MODELO PRINCIPAL (elige uno)
# =============================================================================
echo ""
echo "=== [1/4] Modelo principal LTX-2.3 ==="

if [ "$MODE" = "--full" ]; then
    # Dev model (mayor calidad, más lento, necesita ~80GB VRAM sin fp8)
    hf_download "Lightricks/LTX-2.3" "ltx-2.3-22b-dev.safetensors" "$MODELS_DIR"
fi

# Distilled model (recomendado: 8 pasos, mucho más rápido)
hf_download "Lightricks/LTX-2.3" "ltx-2.3-22b-distilled-1.1.safetensors" "$MODELS_DIR"

# =============================================================================
# SPATIAL UPSCALER (necesario para pipelines de 2 etapas)
# =============================================================================
echo ""
echo "=== [2/4] Spatial Upscaler ==="
hf_download "Lightricks/LTX-2.3" "ltx-2.3-spatial-upscaler-x2-1.1.safetensors" "$MODELS_DIR"

if [ "$MODE" = "--full" ]; then
    hf_download "Lightricks/LTX-2.3" "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors" "$MODELS_DIR"
fi

# =============================================================================
# DISTILLED LORA (necesario para pipelines de 2 etapas excepto DistilledPipeline)
# =============================================================================
echo ""
echo "=== [3/4] Distilled LoRA ==="
hf_download "Lightricks/LTX-2.3" "ltx-2.3-22b-distilled-lora-384-1.1.safetensors" "$MODELS_DIR"

# =============================================================================
# GEMMA TEXT ENCODER
# =============================================================================
echo ""
echo "=== [4/4] Gemma 3 Text Encoder ==="
GEMMA_DIR="$MODELS_DIR/gemma-3-12b"
mkdir -p "$GEMMA_DIR"
# Nota: necesitas aceptar los terminos de Google en HuggingFace primero:
# https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir "$GEMMA_DIR" \
    --local-dir-use-symlinks False

# =============================================================================
# LORAS OPCIONALES (solo --full)
# =============================================================================
if [ "$MODE" = "--full" ]; then
    echo ""
    echo "=== [Extra] LoRAs adicionales ==="
    LORA_DIR="$MODELS_DIR/loras"
    mkdir -p "$LORA_DIR"

    hf_download "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control" \
        "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" "$LORA_DIR"

    hf_download "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control" \
        "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors" "$LORA_DIR"
fi

echo ""
echo "=========================================="
echo " Modelos descargados en: $MODELS_DIR"
echo " Siguiente paso: bash scripts/webui.sh"
echo "  o directamente: source .venv/bin/activate && python scripts/webui.py"
echo "=========================================="
