"""
main.py — UGC Video Ad Generator (Production-Grade)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run: python main.py
UI:  http://localhost:7860

Setup:
  1. pip install -r requirements.txt
  2. Set LIVE_PORTRAIT_DIR env var → path to cloned LivePortrait repo
  3. (Optional) Set ELEVENLABS_API_KEY for premium voice quality
  4. (Optional) Set REALESRGAN_WEIGHTS for 4K-quality upscaling
"""

import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from pipeline import (
    ALL_VOICES, ASPECT_RATIOS, COLOR_GRADES, GRAIN_LEVELS, CAPTION_STYLES,
    generate_audio, get_audio_duration,
    merge_images, validate_and_preprocess_face,
    animate_video,
    post_process, upscale_with_realesrgan,
    generate_caption_filter, generate_hook_overlay,
)

# ── Load .env file ────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ugc")

# ── Config ────────────────────────────────────────────────────────────────────
LIVE_PORTRAIT_DIR = Path(
    os.environ.get("LIVE_PORTRAIT_DIR", "./LivePortrait")
).resolve()

REALESRGAN_WEIGHTS = os.environ.get(
    "REALESRGAN_WEIGHTS", "./weights/RealESRGAN_x4plus.pth"
)

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

def run_pipeline(
    # Inputs
    human_image,
    product_image,
    script,
    # Voice
    voice_key,
    speech_rate,
    speech_pitch,
    elevenlabs_key,
    el_stability,
    el_similarity,
    # Image
    placement,
    product_scale,
    enable_bg_blur,
    bg_blur_strength,
    # Animation
    motion_scale,
    enable_eye,
    enable_lip,
    # Layout & grade
    aspect_ratio,
    color_grade,
    grain_level,
    enable_vignette,
    enable_shake,
    shake_intensity,
    # Captions
    caption_style,
    hook_text,
    # Audio
    bgm_file,
    bgm_volume,
    voice_volume,
    # Fade
    fade_in,
    fade_out,
    # Quality
    enable_upscale,
):
    # ── Input validation ─────────────────────────────────────────────────────
    if human_image is None:
        return None, "❌ Upload a base human image."
    if product_image is None:
        return None, "❌ Upload a product image (PNG)."
    if not script.strip():
        return None, "❌ Enter an ad script."

    # Override ElevenLabs key from UI if provided
    if elevenlabs_key and elevenlabs_key.strip():
        os.environ["ELEVENLABS_API_KEY"] = elevenlabs_key.strip()

    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_DIR / f"job_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'═'*60}\n  JOB {job_id}\n{'═'*60}")
    status = []
    t0 = time.time()

    try:
        # ── Phase 2: Voice ───────────────────────────────────────────────────
        yield None, "⏳ Generating voice..."
        audio_path = generate_audio(
            script, voice_key, job_dir,
            rate=speech_rate, pitch=speech_pitch,
            stability=float(el_stability),
            similarity_boost=float(el_similarity),
        )
        duration = get_audio_duration(audio_path)
        status.append(f"✅ Voice generated ({duration:.1f}s)")
        yield None, "\n".join(status)

        # ── Phase 3a: Image merge ─────────────────────────────────────────────
        yield None, "\n".join(status) + "\n⏳ Placing product on image..."
        merged = merge_images(
            human_image, product_image, job_dir,
            placement=placement,
            product_scale=float(product_scale),
            enable_bg_blur=enable_bg_blur,
            bg_blur_strength=int(bg_blur_strength),
        )
        status.append("✅ Product placed on image")
        yield None, "\n".join(status)

        # ── Phase 3b: Face preprocessing ──────────────────────────────────────
        yield None, "\n".join(status) + "\n⏳ Detecting face..."
        face = validate_and_preprocess_face(merged, job_dir)
        status.append("✅ Face detected and preprocessed")
        yield None, "\n".join(status)

        # ── Phase 4: Animation ───────────────────────────────────────────────
        yield None, "\n".join(status) + "\n⏳ Animating face (this takes 30–120s on GPU)..."
        animated = animate_video(
            face, audio_path, job_dir,
            live_portrait_dir=LIVE_PORTRAIT_DIR,
            motion_scale=float(motion_scale),
            enable_eye_retargeting=enable_eye,
            enable_lip_retargeting=enable_lip,
        )
        status.append("✅ Face animated with lip-sync")
        yield None, "\n".join(status)

        # ── Phase 4b: Optional upscale ────────────────────────────────────────
        if enable_upscale:
            yield None, "\n".join(status) + "\n⏳ Upscaling frames with Real-ESRGAN..."
            animated = upscale_with_realesrgan(animated, job_dir, REALESRGAN_WEIGHTS)
            status.append("✅ Frames upscaled (Real-ESRGAN)")
            yield None, "\n".join(status)

        # ── Phase 5: Captions ────────────────────────────────────────────────
        caption_filter = ""
        hook_filter = ""
        if caption_style != "none":
            caption_filter = generate_caption_filter(script, duration, style=caption_style)
            status.append(f"✅ Captions generated ({caption_style} style)")
        if hook_text.strip():
            hook_filter = generate_hook_overlay(hook_text, duration=2.5, style="ugc")
            status.append("✅ Hook text overlay added")

        # ── Phase 5: Post-process ─────────────────────────────────────────────
        yield None, "\n".join(status) + "\n⏳ Rendering final video..."
        bgm_path = bgm_file if bgm_file else None
        final_output = OUTPUT_DIR / f"ugc_ad_{job_id}.mp4"

        final = post_process(
            animated_video=animated,
            audio_path=audio_path,
            job_dir=job_dir,
            output_path=final_output,
            aspect_ratio=aspect_ratio,
            color_grade=color_grade,
            grain_level=grain_level,
            enable_vignette=enable_vignette,
            enable_camera_shake=enable_shake,
            shake_intensity=float(shake_intensity),
            fade_in_duration=float(fade_in),
            fade_out_duration=float(fade_out),
            caption_filter=caption_filter,
            hook_filter=hook_filter,
            bgm_path=bgm_path,
            bgm_volume=float(bgm_volume),
            voice_volume=float(voice_volume),
        )

        elapsed = time.time() - t0
        status.append(f"✅ Final video rendered")
        status.append(f"\n🎬 Done in {elapsed:.0f}s → {final.name}")
        yield str(final), "\n".join(status)

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        log.error(f"Pipeline error: {e}")
        yield None, "\n".join(status) + f"\n\n❌ {type(e).__name__}:\n{e}"

    except Exception as e:
        log.exception("Unexpected error")
        yield None, "\n".join(status) + f"\n\n❌ Unexpected: {type(e).__name__}: {e}"


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui():
    css = """
    footer { display: none !important; }
    .section-title { font-weight: 700; font-size: 15px; margin-top: 8px; color: #f97316; }
    .status-log { font-family: 'Courier New', monospace !important; font-size: 12px; }
    """

    with gr.Blocks(
        title="UGC Ad Generator",
        theme=gr.themes.Default(
            primary_hue="orange",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css=css,
    ) as demo:

        gr.Markdown("""
        # 🎬 UGC Video Ad Generator
        **Realistic • Lip-synced • Cinematic** — Upload face + product + script → ready-to-post UGC ad
        """)

        with gr.Row(equal_height=False):

            # ════════════════════════════════════════════
            # LEFT COLUMN — All Controls
            # ════════════════════════════════════════════
            with gr.Column(scale=1, min_width=400):

                # ── Inputs ───────────────────────────────
                gr.Markdown("### 📥 Inputs")
                human_image = gr.Image(
                    label="Base Human Image (front-facing, well-lit)",
                    type="filepath", height=220,
                )
                product_image = gr.Image(
                    label="Product Image (PNG with transparent background)",
                    type="filepath", height=180,
                )
                script_input = gr.Textbox(
                    label="Ad Script",
                    placeholder="e.g. I've been using this for 30 days and my skin has completely transformed. No harsh chemicals, no irritation — just real results. Link in bio.",
                    lines=4, max_lines=8,
                )

                # ── Voice Settings ────────────────────────
                with gr.Accordion("🎙️ Voice Settings", open=True):
                    with gr.Row():
                        voice_select = gr.Dropdown(
                            label="Voice",
                            choices=list(ALL_VOICES.keys()),
                            value="Aria (Female, US)",
                        )
                        speech_rate = gr.Dropdown(
                            label="Speed",
                            choices=["-20%", "-10%", "+0%", "+10%", "+20%"],
                            value="+0%",
                        )
                    speech_pitch = gr.Dropdown(
                        label="Pitch (edge-tts only)",
                        choices=["-10Hz", "+0Hz", "+10Hz", "+20Hz"],
                        value="+0Hz",
                    )
                    gr.Markdown("**ElevenLabs (optional — better quality)**")
                    elevenlabs_key = gr.Textbox(
                        label="ElevenLabs API Key",
                        placeholder="sk-... (leave empty to use free edge-tts)",
                        type="password",
                    )
                    with gr.Row():
                        el_stability = gr.Slider(
                            label="Stability", minimum=0.0, maximum=1.0,
                            step=0.05, value=0.50,
                        )
                        el_similarity = gr.Slider(
                            label="Similarity Boost", minimum=0.0, maximum=1.0,
                            step=0.05, value=0.75,
                        )

                # ── Product Placement ─────────────────────
                with gr.Accordion("📦 Product Placement", open=True):
                    with gr.Row():
                        placement = gr.Dropdown(
                            label="Position",
                            choices=["bottom_right", "bottom_left", "top_right",
                                     "top_left", "center", "hand"],
                            value="bottom_right",
                        )
                        product_scale = gr.Slider(
                            label="Size (% of frame width)",
                            minimum=0.10, maximum=0.55, step=0.05, value=0.28,
                        )
                    enable_bg_blur = gr.Checkbox(
                        label="Portrait Mode Blur (background blur)",
                        value=False,
                    )
                    bg_blur_strength = gr.Slider(
                        label="Blur Strength", minimum=2, maximum=20,
                        step=1, value=8, visible=False,
                    )
                    enable_bg_blur.change(
                        fn=lambda x: gr.update(visible=x),
                        inputs=enable_bg_blur,
                        outputs=bg_blur_strength,
                    )

                # ── Animation ─────────────────────────────
                with gr.Accordion("🎭 Animation (LivePortrait)", open=False):
                    motion_scale = gr.Slider(
                        label="Head Movement Intensity",
                        minimum=0.3, maximum=2.0, step=0.1, value=1.0,
                    )
                    with gr.Row():
                        enable_eye = gr.Checkbox(label="Eye Blink Retargeting", value=True)
                        enable_lip = gr.Checkbox(label="Lip Sync Retargeting", value=True)

                # ── Visual Effects ────────────────────────
                with gr.Accordion("🎨 Visual Effects", open=True):
                    with gr.Row():
                        aspect_ratio = gr.Dropdown(
                            label="Aspect Ratio",
                            choices=list(ASPECT_RATIOS.keys()),
                            value="9:16 Portrait (TikTok/Reels)",
                        )
                        color_grade = gr.Dropdown(
                            label="Color Grade",
                            choices=list(COLOR_GRADES.keys()),
                            value="warm",
                        )
                    with gr.Row():
                        grain_level = gr.Dropdown(
                            label="Film Grain (phone-shot look)",
                            choices=list(GRAIN_LEVELS.keys()),
                            value="subtle",
                        )
                    with gr.Row():
                        enable_vignette = gr.Checkbox(label="Vignette", value=True)
                        enable_shake = gr.Checkbox(label="Handheld Shake", value=True)
                    shake_intensity = gr.Slider(
                        label="Shake Intensity",
                        minimum=0.2, maximum=3.0, step=0.1, value=1.0,
                    )
                    with gr.Row():
                        fade_in = gr.Slider(
                            label="Fade In (sec)", minimum=0, maximum=1.5,
                            step=0.1, value=0.4,
                        )
                        fade_out = gr.Slider(
                            label="Fade Out (sec)", minimum=0, maximum=1.5,
                            step=0.1, value=0.5,
                        )

                # ── Captions ─────────────────────────────
                with gr.Accordion("💬 Captions & Hook Text", open=True):
                    caption_style = gr.Dropdown(
                        label="Caption Style",
                        choices=list(CAPTION_STYLES.keys()),
                        value="ugc",
                    )
                    hook_text = gr.Textbox(
                        label="Hook Text (shown first 2.5 seconds)",
                        placeholder="e.g. Wait... this actually works?",
                        lines=1,
                    )

                # ── Background Music ──────────────────────
                with gr.Accordion("🎵 Background Music", open=False):
                    bgm_file = gr.File(
                        label="BGM Audio File (mp3/wav — optional)",
                        file_types=["audio"],
                    )
                    with gr.Row():
                        bgm_volume = gr.Slider(
                            label="BGM Volume", minimum=0.0, maximum=0.5,
                            step=0.01, value=0.12,
                        )
                        voice_volume = gr.Slider(
                            label="Voice Volume", minimum=0.5, maximum=2.0,
                            step=0.05, value=1.0,
                        )

                # ── Quality ───────────────────────────────
                with gr.Accordion("⚡ Quality", open=False):
                    enable_upscale = gr.Checkbox(
                        label="Real-ESRGAN Upscaling (sharper faces — needs GPU + weights)",
                        value=False,
                    )

                generate_btn = gr.Button(
                    "🎬 Generate UGC Ad",
                    variant="primary",
                    size="lg",
                )

            # ════════════════════════════════════════════
            # RIGHT COLUMN — Output
            # ════════════════════════════════════════════
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("### 📤 Output")

                output_video = gr.Video(
                    label="Generated UGC Video Ad",
                    height=520,
                )
                status_box = gr.Textbox(
                    label="Pipeline Status",
                    lines=12,
                    interactive=False,
                    elem_classes=["status-log"],
                    placeholder=(
                        "Pipeline status will appear here...\n\n"
                        "Typical times (with GPU):\n"
                        "  Voice:      2–5s\n"
                        "  Image:      1–2s\n"
                        "  Animation:  30–90s\n"
                        "  Render:     10–20s"
                    ),
                )

                gr.Markdown("""
                ---
                **Tips for best results:**
                - Use a front-facing, well-lit photo (AI-generated works great)
                - Product PNG must have transparent background
                - Keep script under 45 seconds for best lip-sync quality
                - "Warm" color grade + "subtle" grain = most realistic phone-shot look
                """)

        # ── Event binding ─────────────────────────────────────────────────────
        all_inputs = [
            human_image, product_image, script_input,
            voice_select, speech_rate, speech_pitch,
            elevenlabs_key, el_stability, el_similarity,
            placement, product_scale, enable_bg_blur, bg_blur_strength,
            motion_scale, enable_eye, enable_lip,
            aspect_ratio, color_grade, grain_level,
            enable_vignette, enable_shake, shake_intensity,
            caption_style, hook_text,
            bgm_file, bgm_volume, voice_volume,
            fade_in, fade_out,
            enable_upscale,
        ]

        generate_btn.click(
            fn=run_pipeline,
            inputs=all_inputs,
            outputs=[output_video, status_box],
            show_progress="full",
            queue=True,
        )

    return demo


# ── Environment check ─────────────────────────────────────────────────────────

def check_environment():
    log.info("[ENV] Pre-flight checks...")
    ok = True

    if shutil.which("ffmpeg") is None:
        log.warning("⚠️  FFmpeg not found — install: sudo apt install ffmpeg")
        ok = False
    else:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        log.info(f"✅ FFmpeg: {r.stdout.split(chr(10))[0]}")

    if not LIVE_PORTRAIT_DIR.exists():
        log.warning(f"⚠️  LivePortrait not found: {LIVE_PORTRAIT_DIR}")
        log.warning("   Fix: git clone https://github.com/KwaiVGI/LivePortrait.git")
        log.warning(f"   Then: export LIVE_PORTRAIT_DIR={LIVE_PORTRAIT_DIR}")
    else:
        log.info(f"✅ LivePortrait: {LIVE_PORTRAIT_DIR}")

    if os.environ.get("ELEVENLABS_API_KEY"):
        log.info("✅ ElevenLabs API key found")
    else:
        log.info("ℹ️  No ElevenLabs key — using edge-tts (free)")

    try:
        import torch
        if torch.cuda.is_available():
            log.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            log.warning("⚠️  No CUDA GPU — LivePortrait will be slow on CPU")
    except ImportError:
        log.warning("⚠️  PyTorch not installed")

    log.info("[ENV] Done.\n")
    return ok


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_environment()
    demo = build_ui()
    demo.queue(max_size=3)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,       # set True for public gradio.live URL (Colab)
        inbrowser=True,
        show_error=True,
    )
