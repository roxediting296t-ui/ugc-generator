%%writefile /content/ugc-generator/main.py

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

load_dotenv()

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ugc")

LIVE_PORTRAIT_DIR = Path(
    os.environ.get("LIVE_PORTRAIT_DIR", "./LivePortrait")).resolve()
REALESRGAN_WEIGHTS = os.environ.get(
    "REALESRGAN_WEIGHTS", "./weights/RealESRGAN_x4plus.pth")
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASPECT_RATIOS = {
    "9:16 Portrait (TikTok/Reels)": (1080, 1920),
    "1:1 Square (Instagram Feed)":  (1080, 1080),
    "16:9 Landscape (YouTube)":     (1920, 1080),
}
COLOR_GRADES = {
    "warm":      "eq=brightness=0.03:contrast=1.10:saturation=1.15",
    "cool":      "eq=brightness=0.02:contrast=1.08:saturation=1.10",
    "neutral":   "eq=brightness=0.01:contrast=1.05:saturation=1.05",
    "cinematic": "eq=brightness=-0.02:contrast=1.15:saturation=0.90",
    "vibrant":   "eq=brightness=0.04:contrast=1.12:saturation=1.35",
}
TTS_VOICES = {
    "Aria (Female, US)":      "en-US-AriaNeural",
    "Jenny (Female, US)":     "en-US-JennyNeural",
    "Christopher (Male, US)": "en-US-ChristopherNeural",
    "Eric (Male, US)":        "en-US-EricNeural",
    "Guy (Male, US)":         "en-US-GuyNeural",
    "Sonia (Female, UK)":     "en-GB-SoniaNeural",
}


def run_pipeline(
    human_image, product_image, script,
    voice_key, speech_rate,
    placement, product_scale,
    aspect_ratio, color_grade,
    enable_vignette, enable_shake,
    caption_style, hook_text,
    bgm_volume, voice_volume,
    fade_in, fade_out,
):
    # ── Validation ────────────────────────────────────────────────────
    if human_image is None:
        return None, "❌ Upload a human image."
    if product_image is None:
        return None, "❌ Upload a product image."
    if not script.strip():
        return None, "❌ Enter a script."

    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_DIR / f"job_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    status = []
    t0 = time.time()

    try:
        # ── Phase 2: TTS ──────────────────────────────────────────────
        import asyncio, edge_tts
        voice_id = TTS_VOICES.get(voice_key, "en-US-AriaNeural")
        audio_path = job_dir / "voiceover.mp3"

        async def do_tts():
            comm = edge_tts.Communicate(
                text=script, voice=voice_id, rate=speech_rate)
            await comm.save(str(audio_path))

        asyncio.run(do_tts())

        if not audio_path.exists() or audio_path.stat().st_size < 500:
            return None, "❌ TTS failed — check internet connection."

        status.append("✅ Voice generated")

        # ── Get audio duration ────────────────────────────────────────
        import json
        r = subprocess.run(
            ["ffprobe","-v","quiet","-print_format","json",
             "-show_format", str(audio_path)],
            capture_output=True, text=True)
        duration = float(json.loads(r.stdout)["format"]["duration"])

        # ── Phase 3: Image merge ──────────────────────────────────────
        from PIL import Image, ImageFilter
        import numpy as np

        base = Image.open(human_image).convert("RGBA")
        product = Image.open(product_image).convert("RGBA")
        base_w, base_h = base.size
        new_w = int(base_w * float(product_scale))
        ratio = new_w / product.width
        new_h = int(product.height * ratio)
        product = product.resize((new_w, new_h), Image.LANCZOS)
        margin = int(base_w * 0.05)
        prod_w, prod_h = product.size

        pos_map = {
            "bottom_right": (base_w-prod_w-margin, base_h-prod_h-margin),
            "bottom_left":  (margin, base_h-prod_h-margin),
            "top_right":    (base_w-prod_w-margin, margin),
            "top_left":     (margin, margin),
            "center":       (base_w//2-prod_w//2, base_h//2-prod_h//2),
            "hand":         (base_w//2-prod_w//2, int(base_h*0.62)),
        }
        px, py = pos_map.get(placement, pos_map["bottom_right"])
        layer = Image.new("RGBA", base.size, (0,0,0,0))
        layer.paste(product, (px, py), product)
        merged = Image.alpha_composite(base, layer)
        merged_path = job_dir / "merged.png"
        merged.convert("RGB").save(str(merged_path))
        status.append("✅ Product placed on image")

        # ── Phase 3b: Face detect ─────────────────────────────────────
        import cv2
        img_cv = cv2.imread(str(merged_path))
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, 1.05, 4, minSize=(60,60))

        if len(faces) == 0:
            return None, "❌ No face detected. Use a clear front-facing photo."

        x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
        pad = int(max(w,h)*0.45)
        ih,iw = img_cv.shape[:2]
        x1,y1 = max(0,x-pad), max(0,y-pad)
        x2,y2 = min(iw,x+w+pad), min(ih,y+h+pad)
        crop = img_cv[y1:y2, x1:x2]
        sz = max(crop.shape[:2])
        sq = np.zeros((sz,sz,3), dtype=np.uint8)
        yo = (sz-crop.shape[0])//2
        xo = (sz-crop.shape[1])//2
        sq[yo:yo+crop.shape[0], xo:xo+crop.shape[1]] = crop
        face_img = cv2.resize(sq, (512,512), interpolation=cv2.INTER_LANCZOS4)
        face_path = job_dir / "face.png"
        cv2.imwrite(str(face_path), face_img)
        status.append("✅ Face detected")

        # ── Phase 4: LivePortrait ─────────────────────────────────────
        lp_script = LIVE_PORTRAIT_DIR / "inference.py"
        if not lp_script.exists():
            return None, f"❌ LivePortrait not found: {LIVE_PORTRAIT_DIR}"

        cmd = [
            sys.executable, str(lp_script),
            "--source_image", str(face_path),
            "--audio", str(audio_path),
            "--output_dir", str(job_dir),
            "--output_name", "animated",
            "--flag_relative", "--flag_pasteback", "--flag_do_crop",
        ]
        result = subprocess.run(
            cmd, cwd=str(LIVE_PORTRAIT_DIR),
            capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            return None, f"❌ LivePortrait failed:\n{result.stderr[-1500:]}"

        anim_path = job_dir / "animated.mp4"
        if not anim_path.exists():
            candidates = list(job_dir.glob("*.mp4"))
            if not candidates:
                return None, "❌ LivePortrait produced no video output."
            anim_path = candidates[0]
        status.append("✅ Face animated")

        # ── Phase 5: FFmpeg post-process ──────────────────────────────
        tw, th = ASPECT_RATIOS.get(aspect_ratio, (1080,1920))
        grade = COLOR_GRADES.get(color_grade, COLOR_GRADES["warm"])

        vf_parts = [
            f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
            f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1",
            grade,
            "noise=alls=12:allf=t+u",
        ]
        if enable_vignette:
            vf_parts.append("vignette=angle=PI/4")
        if enable_shake:
            vf_parts.append(
                f"crop=w={tw}:h={th}:"
                f"x='{tw}*0.01*sin(t*2.1)':"
                f"y='{th}*0.008*sin(t*1.7)',"
                f"scale={tw}:{th}"
            )
        if fade_in > 0:
            vf_parts.append(f"fade=t=in:st=0:d={fade_in}")
        if fade_out > 0:
            fo_start = max(0, duration - fade_out)
            vf_parts.append(f"fade=t=out:st={fo_start:.2f}:d={fade_out}")

        # Captions
        if caption_style != "none" and script.strip():
            words = script.replace("'","\u2019").split()
            seg_size = 3
            segs = [" ".join(words[i:i+seg_size])
                    for i in range(0,len(words),seg_size)]
            total_w = sum(len(s) for s in segs)
            t_cur = 0.1
            cap_filters = []
            for seg in segs:
                seg_dur = max(0.5, min(3.5, (len(seg)/total_w)*(duration-0.2)))
                safe = seg.replace(":","\:").replace("%","\%")
                cap_filters.append(
                    f"drawtext=text='{safe}':fontsize=72:fontcolor=white:"
                    f"bordercolor=black:borderw=4:x=(w-text_w)/2:y=h-h/6:"
                    f"enable='between(t,{t_cur:.2f},{t_cur+seg_dur:.2f})'"
                )
                t_cur += seg_dur
            vf_parts.extend(cap_filters)

        # Hook text
        if hook_text.strip():
            import re
            safe_hook = re.sub(r'[^\x00-\x7F]','', hook_text).strip()
            safe_hook = safe_hook.replace(":","\:").replace("%","\%")
            if safe_hook:
                vf_parts.append(
                    f"drawtext=text='{safe_hook}':fontsize=80:"
                    f"fontcolor=white:bordercolor=black:borderw=5:"
                    f"x=(w-text_w)/2:y=h/4:"
                    f"enable='between(t,0,2.5)'"
                )

        vf_string = ",".join(vf_parts)
        final_path = OUTPUT_DIR / f"ugc_ad_{job_id}.mp4"

        ff_cmd = [
            "ffmpeg", "-y",
            "-i", str(anim_path),
            "-i", str(audio_path),
            "-map","0:v:0","-map","1:a:0",
            "-vf", vf_string,
            "-c:v","libx264","-crf","16","-preset","fast",
            "-pix_fmt","yuv420p",
            "-c:a","aac","-b:a","192k","-ar","48000",
            "-shortest","-movflags","+faststart",
            str(final_path),
        ]

        ff = subprocess.run(ff_cmd, capture_output=True, text=True, timeout=300)
        if ff.returncode != 0:
            return None, f"❌ FFmpeg failed:\n{ff.stderr[-1500:]}"

        elapsed = time.time()-t0
        status.append(f"✅ Video rendered")
        status.append(f"\n🎬 Done in {elapsed:.0f}s!")
        return str(final_path), "\n".join(status)

    except Exception as e:
        log.exception("Pipeline error")
        return None, f"❌ Error: {type(e).__name__}: {e}"


def build_ui():
    with gr.Blocks(
        title="UGC Ad Generator",
        theme=gr.themes.Default(primary_hue="orange"),
    ) as demo:
        gr.Markdown("# 🎬 UGC Video Ad Generator\n**Realistic • Lip-synced • Cinematic**")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Inputs")
                human_image = gr.Image(
                    label="Human Image (front-facing)",
                    type="filepath", height=220)
                product_image = gr.Image(
                    label="Product Image (PNG)",
                    type="filepath", height=180)
                script_input = gr.Textbox(
                    label="Ad Script", lines=4,
                    placeholder="e.g. This product changed my life in 30 days...")

                with gr.Accordion("🎙️ Voice", open=True):
                    with gr.Row():
                        voice_select = gr.Dropdown(
                            label="Voice",
                            choices=list(TTS_VOICES.keys()),
                            value="Aria (Female, US)")
                        speech_rate = gr.Dropdown(
                            label="Speed",
                            choices=["-20%","-10%","+0%","+10%","+20%"],
                            value="+0%")

                with gr.Accordion("📦 Product", open=True):
                    with gr.Row():
                        placement = gr.Dropdown(
                            label="Position",
                            choices=["bottom_right","bottom_left",
                                     "top_right","top_left","center","hand"],
                            value="bottom_right")
                        product_scale = gr.Slider(
                            label="Size", minimum=0.10,
                            maximum=0.55, step=0.05, value=0.28)

                with gr.Accordion("🎨 Visual Effects", open=True):
                    with gr.Row():
                        aspect_ratio = gr.Dropdown(
                            label="Aspect Ratio",
                            choices=list(ASPECT_RATIOS.keys()),
                            value="9:16 Portrait (TikTok/Reels)")
                        color_grade = gr.Dropdown(
                            label="Color Grade",
                            choices=list(COLOR_GRADES.keys()),
                            value="warm")
                    with gr.Row():
                        enable_vignette = gr.Checkbox(
                            label="Vignette", value=True)
                        enable_shake = gr.Checkbox(
                            label="Handheld Shake", value=True)
                    with gr.Row():
                        fade_in = gr.Slider(
                            label="Fade In (sec)",
                            minimum=0, maximum=1.5, step=0.1, value=0.4)
                        fade_out = gr.Slider(
                            label="Fade Out (sec)",
                            minimum=0, maximum=1.5, step=0.1, value=0.5)

                with gr.Accordion("💬 Captions", open=True):
                    caption_style = gr.Dropdown(
                        label="Caption Style",
                        choices=["ugc","subtitle","none"],
                        value="ugc")
                    hook_text = gr.Textbox(
                        label="Hook Text (first 2.5 seconds)",
                        placeholder="Wait... this actually works?",
                        lines=1)

                with gr.Accordion("🎵 Audio Mix", open=False):
                    with gr.Row():
                        bgm_volume = gr.Slider(
                            label="BGM Volume",
                            minimum=0.0, maximum=0.5,
                            step=0.01, value=0.12)
                        voice_volume = gr.Slider(
                            label="Voice Volume",
                            minimum=0.5, maximum=2.0,
                            step=0.05, value=1.0)

                generate_btn = gr.Button(
                    "🎬 Generate UGC Ad",
                    variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### 📤 Output")
                output_video = gr.Video(
                    label="Generated UGC Video", height=500)
                status_box = gr.Textbox(
                    label="Status", lines=10, interactive=False,
                    placeholder="Status appears here after generation...")

        generate_btn.click(
            fn=run_pipeline,
            inputs=[
                human_image, product_image, script_input,
                voice_select, speech_rate,
                placement, product_scale,
                aspect_ratio, color_grade,
                enable_vignette, enable_shake,
                caption_style, hook_text,
                bgm_volume, voice_volume,
                fade_in, fade_out,
            ],
            outputs=[output_video, status_box],
            queue=True,
        )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
