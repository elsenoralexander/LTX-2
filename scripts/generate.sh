#!/bin/bash
# =============================================================================
# LTX-2.3 — CLI rápido para generar un video desde terminal
#
# Uso:
#   bash scripts/generate.sh "Una persona camina por un bosque brumoso al amanecer"
#   bash scripts/generate.sh "..." --width 1280 --height 720 --num-frames 97
# =============================================================================
set -e

REPO_DIR="${REPO_DIR:-/workspace/LTX-2}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"

PROMPT="${1:-A cinematic shot of a mountain landscape at golden hour, camera slowly panning right}"
shift 2>/dev/null || true   # resto de args pasan directo al pipeline

# Rutas de modelos
CHECKPOINT="$MODELS_DIR/ltx-2.3-22b-distilled-1.1.safetensors"
UPSCALER="$MODELS_DIR/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
DISTILLED_LORA="$MODELS_DIR/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
GEMMA_ROOT="$MODELS_DIR/gemma-3-12b"
OUTPUT="output_$(date +%Y%m%d_%H%M%S).mp4"

# Usar dev model si existe y el distilled no
if [ ! -f "$CHECKPOINT" ]; then
    CHECKPOINT="$MODELS_DIR/ltx-2.3-22b-dev.safetensors"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: No se encontró ningún checkpoint en $MODELS_DIR"
    echo "Ejecuta primero: bash scripts/download_models.sh"
    exit 1
fi

cd "$REPO_DIR"
source .venv/bin/activate

echo "=== Generando video ==="
echo "  Prompt: $PROMPT"
echo "  Output: $OUTPUT"
echo "  Checkpoint: $CHECKPOINT"
echo ""

python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path "$CHECKPOINT" \
    --distilled-lora "$DISTILLED_LORA" 0.8 \
    --spatial-upsampler-path "$UPSCALER" \
    --gemma-root "$GEMMA_ROOT" \
    --prompt "$PROMPT" \
    --output-path "$OUTPUT" \
    --quantization fp8-cast \
    --width 768 \
    --height 512 \
    --num-frames 97 \
    "$@"

echo ""
echo "Video guardado en: $REPO_DIR/$OUTPUT"
