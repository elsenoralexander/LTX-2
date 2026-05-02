"""
LTX-2.3 Web UI — Gradio interface for RunPod

Variables de entorno opcionales:
    MODELS_DIR   ruta a los modelos (default: /workspace/models)
    PORT         puerto del servidor  (default: 7860)
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio"])
    import gradio as gr

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/workspace/models"))
REPO_DIR = Path(os.environ.get("REPO_DIR", "/workspace/LTX-2"))
PORT = int(os.environ.get("PORT", 7860))

DISTILLED_CKPT = MODELS_DIR / "ltx-2.3-22b-distilled-1.1.safetensors"
DEV_CKPT       = MODELS_DIR / "ltx-2.3-22b-dev.safetensors"
UPSCALER_X2    = MODELS_DIR / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
DISTILLED_LORA = MODELS_DIR / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
GEMMA_ROOT     = MODELS_DIR / "gemma-3-12b"

# Aspect ratio → (width, height) — todos múltiplos de 64
RESOLUTION_MAP = {
    ("16:9 — Horizontal", "Rápido"):   (640, 384),
    ("16:9 — Horizontal", "Estándar"): (768, 448),
    ("16:9 — Horizontal", "HD"):       (1024, 576),
    ("4:3 — Clásico",     "Rápido"):   (640, 512),
    ("4:3 — Clásico",     "Estándar"): (768, 576),
    ("4:3 — Clásico",     "HD"):       (1024, 768),
    ("9:16 — Vertical",   "Rápido"):   (384, 640),
    ("9:16 — Vertical",   "Estándar"): (448, 768),
    ("9:16 — Vertical",   "HD"):       (576, 1024),
    ("1:1 — Cuadrado",    "Rápido"):   (512, 512),
    ("1:1 — Cuadrado",    "Estándar"): (640, 640),
    ("1:1 — Cuadrado",    "HD"):       (768, 768),
}

IMAGE_POSITION_LABELS = ["Inicio (frame 0)", "Mitad", "Final"]


def seconds_to_frames(seconds: float) -> int:
    raw = round(seconds * 24)
    raw = max(9, raw)
    n = max(1, round((raw - 1) / 8))
    return min(n * 8 + 1, 257)


def get_checkpoint() -> Path:
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
    aspect_ratio: str,
    quality: str,
    duration_seconds: float,
    pipeline_type: str,
    quantization: str,
    seed: int,
    # Tres imágenes de referencia con su posición y fuerza
    img1_path: str | None,
    img1_position: str,
    img1_strength: float,
    img2_path: str | None,
    img2_position: str,
    img2_strength: float,
    img3_path: str | None,
    img3_position: str,
    img3_strength: float,
) -> tuple[str, str]:

    width, height = RESOLUTION_MAP.get((aspect_ratio, quality), (768, 448))
    num_frames = seconds_to_frames(duration_seconds)

    try:
        checkpoint = get_checkpoint()
    except FileNotFoundError as e:
        return None, str(e)

    pipeline_map = {
        "DistilledPipeline (más rápido)":          "ltx_pipelines.distilled",
        "TI2VidTwoStagesPipeline (recomendado)":   "ltx_pipelines.ti2vid_two_stages",
        "TI2VidTwoStagesHQPipeline (mejor calidad)": "ltx_pipelines.ti2vid_two_stages_hq",
        "TI2VidOneStagePipeline (rápido/prototipo)": "ltx_pipelines.ti2vid_one_stage",
    }
    module = pipeline_map.get(pipeline_type, "ltx_pipelines.distilled")

    cmd = [
        sys.executable, "-m", module,
        "--checkpoint-path", str(checkpoint),
        "--gemma-root", str(GEMMA_ROOT),
        "--prompt", prompt,
        "--output-path", tempfile.mktemp(suffix=".mp4", dir="/tmp"),
        "--width", str(width),
        "--height", str(height),
        "--num-frames", str(num_frames),
        "--seed", str(seed),
    ]

    # Guardar la ruta de salida para devolverla
    output_path = cmd[cmd.index("--output-path") + 1]

    if negative_prompt.strip():
        cmd += ["--negative-prompt", negative_prompt]

    if "distilled" not in module.split(".")[-1]:
        if UPSCALER_X2.exists():
            cmd += ["--spatial-upsampler-path", str(UPSCALER_X2)]
        if DISTILLED_LORA.exists():
            cmd += ["--distilled-lora", str(DISTILLED_LORA), "0.8"]

    # Añadir imágenes de referencia
    position_to_frame = {
        "Inicio (frame 0)": 0,
        "Mitad":            num_frames // 2,
        "Final":            num_frames - 1,
    }

    images = [
        (img1_path, img1_position, img1_strength),
        (img2_path, img2_position, img2_strength),
        (img3_path, img3_position, img3_strength),
    ]
    image_log = []
    for path, position, strength in images:
        if path:
            frame_idx = position_to_frame.get(position, 0)
            cmd += ["--image", path, str(frame_idx), str(round(strength, 2))]
            image_log.append(f"  · {Path(path).name} → frame {frame_idx}, strength {strength:.2f}")

    if quantization != "ninguna":
        cmd += ["--quantization", quantization]

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(REPO_DIR / "packages/ltx-pipelines/src") + ":" +
        str(REPO_DIR / "packages/ltx-core/src") + ":" +
        env.get("PYTHONPATH", "")
    )

    log_lines = [
        f"Resolución: {width}x{height} ({aspect_ratio} · {quality})",
        f"Duración: {duration_seconds}s → {num_frames} frames",
        f"Pipeline: {module}",
    ]
    if image_log:
        log_lines.append("Referencias visuales:")
        log_lines += image_log
    log_lines += ["", f"Comando: {' '.join(cmd)}", ""]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO_DIR))
        log_lines.append(result.stdout)
        if result.returncode != 0:
            log_lines += ["=== STDERR ===", result.stderr]
            return None, "\n".join(log_lines)
    except Exception as e:
        log_lines.append(f"Error: {e}")
        return None, "\n".join(log_lines)

    if not Path(output_path).exists():
        return None, "\n".join(log_lines) + "\nError: no se generó el archivo de video."

    log_lines.append(f"\nVideo generado: {output_path}")
    return output_path, "\n".join(log_lines)


# =============================================================================
# UI
# =============================================================================
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="LTX-2.3 — Generador de Video") as demo:
        gr.Markdown("# LTX-2.3 — Generador de Video\nModelo de 22B parámetros corriendo en tu GPU alquilada.")

        with gr.Row():
            # --- Columna izquierda: texto + referencias ---
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

                gr.Markdown("### Referencias visuales (hasta 3)")
                gr.Markdown("*Puedes anclar cada imagen al inicio, la mitad o el final del video.*")

                with gr.Row():
                    with gr.Column():
                        img1 = gr.Image(label="Referencia 1", type="filepath")
                        img1_pos = gr.Dropdown(IMAGE_POSITION_LABELS, value="Inicio (frame 0)", label="Posición", show_label=False)
                        img1_str = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Fuerza")
                    with gr.Column():
                        img2 = gr.Image(label="Referencia 2", type="filepath")
                        img2_pos = gr.Dropdown(IMAGE_POSITION_LABELS, value="Mitad", label="Posición", show_label=False)
                        img2_str = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Fuerza")
                    with gr.Column():
                        img3 = gr.Image(label="Referencia 3", type="filepath")
                        img3_pos = gr.Dropdown(IMAGE_POSITION_LABELS, value="Final", label="Posición", show_label=False)
                        img3_str = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Fuerza")

            # --- Columna derecha: parámetros ---
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
                    info="fp8-cast reduce VRAM ~50%. fp8-scaled-mm solo en H100.",
                )
                aspect_ratio = gr.Dropdown(
                    label="Formato",
                    choices=[
                        "16:9 — Horizontal",
                        "4:3 — Clásico",
                        "9:16 — Vertical",
                        "1:1 — Cuadrado",
                    ],
                    value="16:9 — Horizontal",
                )
                quality = gr.Dropdown(
                    label="Calidad / Resolución",
                    choices=["Rápido", "Estándar", "HD"],
                    value="Estándar",
                    info="Rápido=640px · Estándar=768px · HD=1024px",
                )
                duration_seconds = gr.Slider(
                    minimum=2, maximum=10, value=4, step=0.5,
                    label="Duración (segundos)",
                    info="Se convierte a frames automáticamente (24fps)",
                )
                seed = gr.Number(value=42, label="Seed", precision=0)

        generate_btn = gr.Button("Generar Video", variant="primary", size="lg")

        with gr.Row():
            video_output = gr.Video(label="Video generado")
            log_output = gr.Textbox(label="Log", lines=15, max_lines=30)

        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt, negative_prompt, aspect_ratio, quality,
                duration_seconds, pipeline_type, quantization, seed,
                img1, img1_pos, img1_str,
                img2, img2_pos, img2_str,
                img3, img3_pos, img3_str,
            ],
            outputs=[video_output, log_output],
        )

        gr.Markdown("""
        ### Tips
        - **DistilledPipeline**: ~2-5 min en RTX 4090 · buena calidad
        - **TI2VidTwoStagesPipeline**: ~8-15 min · calidad producción
        - **fp8-cast**: reduce VRAM a ~24 GB (ideal RTX 4090)
        - **Referencia 1 en "Inicio"**: ancla el primer fotograma del video a tu imagen
        - Puedes usar solo una, dos o las tres referencias
        """)

    return demo


if __name__ == "__main__":
    missing = []
    if not DISTILLED_CKPT.exists() and not DEV_CKPT.exists():
        missing.append(f"Checkpoint: {DISTILLED_CKPT} o {DEV_CKPT}")
    if not GEMMA_ROOT.exists():
        missing.append(f"Gemma: {GEMMA_ROOT}")

    if missing:
        print("AVISO: faltan modelos (ejecuta download_models.sh):")
        for m in missing:
            print(f"  - {m}")
        print("Iniciando UI de todas formas.")

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=False)
