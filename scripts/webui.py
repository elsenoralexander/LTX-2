"""
LTX-2.3 Web UI — Gradio interface for RunPod
Corre en el pod y accedes desde tu navegador via el puerto público de RunPod.

Uso:
    source /workspace/LTX-2/.venv/bin/activate
    python /workspace/LTX-2/scripts/webui.py

Variables de entorno opcionales:
    MODELS_DIR   ruta a los modelos (default: /workspace/models)
    PORT         puerto del servidor  (default: 7860)
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# --- Instalar gradio si no está ---
try:
    import gradio as gr
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio"])
    import gradio as gr

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/workspace/models"))
REPO_DIR = Path(os.environ.get("REPO_DIR", "/workspace/LTX-2"))
PORT = int(os.environ.get("PORT", 7860))

# Rutas de modelos (se auto-detectan)
DISTILLED_CKPT = MODELS_DIR / "ltx-2.3-22b-distilled-1.1.safetensors"
DEV_CKPT = MODELS_DIR / "ltx-2.3-22b-dev.safetensors"
UPSCALER_X2 = MODELS_DIR / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
DISTILLED_LORA = MODELS_DIR / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
GEMMA_ROOT = MODELS_DIR / "gemma-3-12b"


def get_checkpoint() -> Path:
    """Devuelve el checkpoint disponible (distilled tiene prioridad)."""
    if DISTILLED_CKPT.exists():
        return DISTILLED_CKPT
    if DEV_CKPT.exists():
        return DEV_CKPT
    raise FileNotFoundError(
        f"No se encontró ningún checkpoint en {MODELS_DIR}.\n"
        "Ejecuta primero: bash scripts/download_models.sh"
    )


def generate_video(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    pipeline_type: str,
    quantization: str,
    seed: int,
    image_path: str | None,
) -> tuple[str, str]:
    """Llama al pipeline de LTX-2 y devuelve (ruta_video, log)."""
    output_path = tempfile.mktemp(suffix=".mp4", dir="/tmp")

    try:
        checkpoint = get_checkpoint()
    except FileNotFoundError as e:
        return None, str(e)

    # Seleccionar pipeline y módulo
    pipeline_map = {
        "DistilledPipeline (más rápido)": "ltx_pipelines.distilled",
        "TI2VidTwoStagesPipeline (recomendado)": "ltx_pipelines.ti2vid_two_stages",
        "TI2VidTwoStagesHQPipeline (mejor calidad)": "ltx_pipelines.ti2vid_two_stages_hq",
        "TI2VidOneStagePipeline (rápido/prototipo)": "ltx_pipelines.ti2vid_one_stage",
    }
    module = pipeline_map.get(pipeline_type, "ltx_pipelines.distilled")

    cmd = [
        sys.executable, "-m", module,
        "--checkpoint-path", str(checkpoint),
        "--gemma-root", str(GEMMA_ROOT),
        "--prompt", prompt,
        "--output-path", output_path,
        "--width", str(width),
        "--height", str(height),
        "--num-frames", str(num_frames),
        "--seed", str(seed),
    ]

    # Negative prompt
    if negative_prompt.strip():
        cmd += ["--negative-prompt", negative_prompt]

    # Upscaler y LoRA (necesarios para pipelines de 2 etapas)
    if "distilled" not in module.split(".")[-1]:
        if UPSCALER_X2.exists():
            cmd += ["--spatial-upsampler-path", str(UPSCALER_X2)]
        if DISTILLED_LORA.exists():
            cmd += ["--distilled-lora", str(DISTILLED_LORA), "0.8"]

    # Imagen de condicionamiento (opcional)
    if image_path:
        cmd += ["--image", image_path, "0", "1.0"]

    # Quantization
    if quantization != "ninguna":
        cmd += ["--quantization", quantization]

    # Ejecutar
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_DIR / "packages/ltx-pipelines/src") + ":" + \
                        str(REPO_DIR / "packages/ltx-core/src") + ":" + \
                        env.get("PYTHONPATH", "")

    log_lines = [f"Ejecutando pipeline: {module}", f"Comando: {' '.join(cmd)}", ""]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO_DIR),
        )
        log_lines.append(result.stdout)
        if result.returncode != 0:
            log_lines.append("=== STDERR ===")
            log_lines.append(result.stderr)
            return None, "\n".join(log_lines)
    except Exception as e:
        log_lines.append(f"Error: {e}")
        return None, "\n".join(log_lines)

    if not Path(output_path).exists():
        return None, "\n".join(log_lines) + "\nError: no se generó el archivo de video."

    log_lines.append(f"\nVideo generado: {output_path}")
    return output_path, "\n".join(log_lines)


# =============================================================================
# Interfaz Gradio
# =============================================================================
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="LTX-2.3 — Generador de Video", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# LTX-2.3 — Generador de Video Local\nModelo de 22B parámetros corriendo en tu GPU alquilada.")

        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A person walks through a foggy forest at dawn, cinematic lighting...",
                    lines=4,
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt (opcional)",
                    placeholder="blurry, low quality, distorted...",
                    lines=2,
                )
                image_input = gr.Image(
                    label="Imagen de condicionamiento (opcional)",
                    type="filepath",
                )

            with gr.Column(scale=1):
                pipeline_type = gr.Dropdown(
                    label="Pipeline",
                    choices=[
                        "DistilledPipeline (más rápido)",
                        "TI2VidTwoStagesPipeline (recomendado)",
                        "TI2VidTwoStagesHQPipeline (mejor calidad)",
                        "TI2VidOneStagePipeline (rápido/prototipo)",
                    ],
                    value="DistilledPipeline (más rápido)",
                )
                quantization = gr.Dropdown(
                    label="Quantization",
                    choices=["ninguna", "fp8-cast", "fp8-scaled-mm"],
                    value="fp8-cast",
                    info="fp8-cast reduce VRAM ~50%. Usar fp8-scaled-mm solo en H100.",
                )
                width = gr.Slider(256, 1280, value=768, step=32, label="Ancho (px)")
                height = gr.Slider(256, 1280, value=512, step=32, label="Alto (px)")
                num_frames = gr.Slider(9, 257, value=97, step=8, label="Frames (múltiplo de 8 + 1)")
                seed = gr.Number(value=42, label="Seed", precision=0)

        generate_btn = gr.Button("Generar Video", variant="primary", size="lg")

        with gr.Row():
            video_output = gr.Video(label="Video generado")
            log_output = gr.Textbox(label="Log", lines=15, max_lines=30)

        generate_btn.click(
            fn=generate_video,
            inputs=[prompt, negative_prompt, width, height, num_frames,
                    pipeline_type, quantization, seed, image_input],
            outputs=[video_output, log_output],
        )

        gr.Markdown("""
        ### Tips
        - **DistilledPipeline**: 8 pasos — más rápido (~2-5 min en A100)
        - **TI2VidTwoStagesPipeline**: calidad producción, más lento (~10-15 min)
        - **fp8-cast**: reduce VRAM a ~24 GB (ideal para RTX 4090 o A100 40GB)
        - Frames recomendados: 97 (~4s a 24fps), 145 (~6s), 193 (~8s)
        """)

    return demo


if __name__ == "__main__":
    # Verificar modelos
    missing = []
    if not DISTILLED_CKPT.exists() and not DEV_CKPT.exists():
        missing.append(f"Checkpoint: {DISTILLED_CKPT} o {DEV_CKPT}")
    if not GEMMA_ROOT.exists():
        missing.append(f"Gemma: {GEMMA_ROOT}")

    if missing:
        print("AVISO: faltan estos modelos (ejecuta download_models.sh):")
        for m in missing:
            print(f"  - {m}")
        print("Iniciando UI de todas formas (generación fallará sin los modelos).")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,   # True si quieres link público de Gradio
    )
