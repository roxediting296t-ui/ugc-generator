"""
UGC Video Ad Generator — app.py
Single file, no external pipeline imports needed.
Run: streamlit run app.py
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter

# ── Config ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ugc")

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LIVE_PORTRAIT_DIR = Path(
    os.environ.get("LIVE_PORTRAIT_DIR", "./LivePortrait")
).resolve()

# ── Constants ─────────────────────────────────────────────────────────────────
TTS_VOICES = {
    "Aria (Female, US — Warm)":       "en-US-AriaNeural",
    "Jenny (Female, US — Friendly)":  "en-US-JennyNeural",
    "Sara (Female, US — Soft)":       "en-US-SaraNeural",
    "Christopher (Male, US — Deep)":  "en-US-ChristopherNeural",
    "Eric (Male, US — Natural)":      "en-US-EricNeural",
    "Guy (Male, US — Confident)":     "en-US-GuyNeural",
    "Sonia (Female, UK)":             "en-GB-SoniaNeural",
    "Ryan (Male, UK)":                "en-GB-RyanNeural",
}

ASPECT_RATIOS = {
    "9:16 Portrait — TikTok/Reels":  (1080, 1920),
    "1:1 Square — Instagram Feed":   (1080, 1080),
    "16:9 Landscape — YouTube":      (1920, 1080),
    "4:5 Portrait — Instagram":      (1080, 1350),
}

COLOR_GRADES = {
    "warm":      "eq=brightness=0.03:contrast=1.10:saturation=1.15:gamma_r=1.06:gamma_b=0.96",
    "cool":      "eq=brightness=0.02:contrast=1.08:saturation=1.10:gamma_r=0.95:gamma_b=1.06",
    "neutral":   "eq=brightness=0.01:contrast=1.05:saturation=1.05",
    "cinematic": "eq=brightness=-0.02:contrast=1.18:saturation=0.88:gamma=1.05",
    "vibrant":   "eq=brightness=0.04:contrast=1.12:saturation=1.40",
    "matte":     "eq=brightness=0.06:contrast=0.94:saturation=0.82:gamma=1.12",
}

PLACEMENTS = [
    "bottom_right", "bottom_left",
    "top_right", "top_left",
    "center", "hand"
]


# ── Helper Functions ──────────────────────────────────────────────────────────

def get_audio_duration(audio_path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", str(audio_path)],
        capture_output=True, text=True
    )
    return float(json.loads(r.stdout)["format"]["duration"])


def generate_voice(script: str, voice_id: str,
                   speed: str, output_path: Path) -> Path:
    import edge_tts

    async def _tts():
        c = edge_tts.Communicate(text=script, voice=voice_id, rate=speed)
        await c.save(str(output_path))

    asyncio.run(_tts())

    if not output_path.exists() or output_path.stat().st_size < 500:
        raise RuntimeError(
            "TTS failed — check internet connection. "
            "edge-tts streams from Microsoft servers."
        )
    return output_path


def place_product(human_path: str, product_path: str,
                  placement: str, scale: float,
                  blur_bg: bool, blur_strength: int,
                  job_dir: Path) -> Path:
    base    = Image.open(human_path).convert("RGBA")
    product = Image.open(product_path).convert("RGBA")

    # Optional background blur — portrait mode feel
    if blur_bg:
        blurred = base.filter(ImageFilter.GaussianBlur(radius=blur_strength))
        bw, bh  = base.size
        mask    = Image.new("L", (bw, bh), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        cx, cy  = bw // 2, bh // 3
        rx, ry  = int(bw * 0.35), int(bh * 0.40)
        for i in range(min(rx, ry), 0, -2):
            alpha = int(255 * (i / min(rx, ry)) ** 0.6)
            draw.ellipse([cx-i, cy-i, cx+i, cy+i], fill=alpha)
        base = Image.composite(base, blurred, mask)

    bw, bh  = base.size
    nw      = int(bw * scale)
    nh      = int(product.height * (nw / product.width))
    product = product.resize((nw, nh), Image.LANCZOS)
    pw, ph  = product.size
    mg      = int(bw * 0.05)

    pos_map = {
        "bottom_right": (bw-pw-mg, bh-ph-mg),
        "bottom_left":  (mg, bh-ph-mg),
        "top_right":    (bw-pw-mg, mg),
        "top_left":     (mg, mg),
        "center":       (bw//2-pw//2, bh//2-ph//2),
        "hand":         (bw//2-pw//2, int(bh*0.62)),
    }
    px, py  = pos_map.get(placement, (bw-pw-mg, bh-ph-mg))

    # Drop shadow
    shadow_color = Image.new("RGBA", product.size, (0, 0, 0, 90))
    shadow       = Image.new("RGBA", product.size, (0, 0, 0, 0))
    shadow.paste(shadow_color, mask=product.split()[-1])
    shadow       = shadow.filter(ImageFilter.GaussianBlur(radius=10))
    shadow_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shadow_layer.paste(shadow, (px+8, py+8), shadow)
    base         = Image.alpha_composite(base, shadow_layer)

    # Product
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(product, (px, py), product)
    merged = Image.alpha_composite(base, layer)

    out = job_dir / "merged.png"
    merged.convert("RGB").save(str(out), "PNG", quality=100)
    return out


def detect_face(image_path: Path, job_dir: Path) -> Path:
    img  = cv2.imread(str(image_path))
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cas  = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cas.detectMultiScale(gray, 1.05, 4, minSize=(60, 60))

    if len(faces) == 0:
        raise ValueError(
            "No face detected.\n"
            "Tips:\n"
            "• Use a clear front-facing photo\n"
            "• Good lighting, no heavy shadows\n"
            "• Face at least 60×60 pixels in the image\n"
            "• AI-generated portraits work great (Midjourney, SDXL)"
        )

    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    pad = int(max(w, h) * 0.45)
    ih, iw = img.shape[:2]
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(iw, x+w+pad), min(ih, y+h+pad)
    crop = img[y1:y2, x1:x2]
    sz   = max(crop.shape[:2])
    sq   = np.zeros((sz, sz, 3), dtype=np.uint8)
    yo   = (sz - crop.shape[0]) // 2
    xo   = (sz - crop.shape[1]) // 2
    sq[yo:yo+crop.shape[0], xo:xo+crop.shape[1]] = crop

    # Sharpen slightly for LivePortrait
    face = cv2.resize(sq, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    kernel = np.array([[0, -0.3, 0], [-0.3, 2.2, -0.3], [0, -0.3, 0]])
    face   = np.clip(cv2.filter2D(face, -1, kernel), 0, 255).astype(np.uint8)

    out = job_dir / "face.png"
    cv2.imwrite(str(out), face)
    return out


def run_liveportrait(face_path: Path, audio_path: Path,
                     job_dir: Path, motion: float) -> Path:
    lp = LIVE_PORTRAIT_DIR / "inference.py"
    if not lp.exists():
        raise FileNotFoundError(
            f"LivePortrait not found: {LIVE_PORTRAIT_DIR}\n"
            "Run Cell 3 to clone it."
        )

    cmd = [
        sys.executable, str(lp),
        "--source_image",  str(face_path),
        "--audio",         str(audio_path),
        "--output_dir",    str(job_dir),
        "--output_name",   "animated",
        "--flag_relative",
        "--flag_pasteback",
        "--flag_do_crop",
    ]
    if motion != 1.0:
        cmd += ["--driving_multiplier", str(motion)]

    res = subprocess.run(
        cmd, cwd=str(LIVE_PORTRAIT_DIR),
        capture_output=True, text=True, timeout=600
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"LivePortrait failed:\n{res.stderr[-2000:]}"
        )

    out = job_dir / "animated.mp4"
    if not out.exists():
        mp4s = list(job_dir.glob("*.mp4"))
        if not mp4s:
            raise FileNotFoundError("LivePortrait produced no video.")
        out = mp4s[0]
    return out


def render_final(
    anim_path: Path, audio_path: Path,
    job_dir: Path, job_id: str,
    duration: float,
    # effects
    tw: int, th: int,
    grade: str,
    vignette: bool, shake: bool, shake_int: float,
    grain: int,
    fade_in: float, fade_out: float,
    # text
    caption_style: str, script: str,
    hook_text: str,
) -> Path:
    vf = [
        f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
        f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1",
        grade,
    ]

    if grain > 0:
        vf.append(f"noise=alls={grain}:allf=t+u")

    if vignette:
        vf.append("vignette=angle=PI/4:mode=forward")

    if shake:
        si = shake_int
        vf.append(
            f"crop=w={tw}:h={th}:"
            f"x='{int(tw*0.012*si)}*sin(t*2.1+0.3)':"
            f"y='{int(th*0.010*si)}*sin(t*1.7+1.1)',"
            f"scale={tw}:{th}"
        )

    if fade_in > 0:
        vf.append(f"fade=t=in:st=0:d={fade_in}:color=black")
    if fade_out > 0:
        vf.append(
            f"fade=t=out:st={max(0,duration-fade_out):.2f}"
            f":d={fade_out}:color=black"
        )

    # Hook text overlay
    if hook_text.strip():
        hs = re.sub(r'[^\x00-\x7F]', '', hook_text.strip())
        hs = hs.replace(":", r"\:").replace("%", r"\%").replace("'", "\u2019")
        if hs:
            fade_expr = (
                f"if(lt(t,0.4),t/0.4,"
                f"if(lt(t,2.1),1,(2.5-t)/0.4))"
            )
            vf.append(
                f"drawtext=text='{hs}'"
                f":fontsize={int(tw*0.072)}"
                f":fontcolor=white"
                f":bordercolor=black:borderw=4"
                f":x=(w-text_w)/2:y=h/5"
                f":alpha='{fade_expr}'"
                f":enable='between(t,0,2.5)'"
            )

    # Captions
    if caption_style != "none" and script.strip():
        words    = script.replace("'", "\u2019").split()
        seg_size = 3 if caption_style == "ugc" else 6
        segs     = [" ".join(words[i:i+seg_size])
                    for i in range(0, len(words), seg_size)]
        tot_w    = sum(len(s) for s in segs)
        tc       = 0.1
        fs       = int(tw * 0.065) if caption_style == "ugc" else int(tw * 0.048)
        y_pos    = "h-h/5" if caption_style == "ugc" else "h-h/8"

        for seg in segs:
            sd   = max(0.5, min(3.8, (len(seg)/tot_w)*(duration-0.2)))
            safe = (seg.replace(":", r"\:")
                       .replace("%", r"\%")
                       .replace("'", "\u2019"))
            vf.append(
                f"drawtext=text='{safe}'"
                f":fontsize={fs}"
                f":fontcolor=white"
                f":bordercolor=black:borderw=4"
                f":x=(w-text_w)/2:y={y_pos}"
                f":enable='between(t,{tc:.2f},{tc+sd:.2f})'"
            )
            tc += sd

    final = OUTPUT_DIR / f"ugc_ad_{job_id}.mp4"
    cmd   = [
        "ffmpeg", "-y",
        "-i", str(anim_path),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", ",".join(vf),
        "-c:v", "libx264", "-crf", "16", "-preset", "fast",
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-shortest", "-movflags", "+faststart",
        str(final),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{r.stderr[-1500:]}")
    return final


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_pipeline(
    human_file, product_file, script,
    voice_key, speed,
    placement, prod_scale,
    blur_bg, blur_strength,
    motion_scale,
    aspect_key, color_key,
    grain_level, vignette, shake, shake_int,
    fade_in, fade_out,
    caption_style, hook_text,
):
    job_id  = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_DIR / f"job_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    status = st.empty()
    prog   = st.progress(0)

    def update(msg, pct):
        status.info(msg)
        prog.progress(pct)

    try:
        # Phase 1: Voice
        update("⏳ Generating voice...", 10)
        audio_path = job_dir / "voice.mp3"
        voice_id   = TTS_VOICES[voice_key]
        generate_voice(script, voice_id, speed, audio_path)
        duration   = get_audio_duration(audio_path)
        update(f"✅ Voice ready ({duration:.1f}s)\n⏳ Processing image...", 25)

        # Phase 2: Image
        human_tmp   = job_dir / "human.png"
        product_tmp = job_dir / "product.png"
        Image.open(human_file).save(str(human_tmp))
        Image.open(product_file).save(str(product_tmp))

        merged = place_product(
            str(human_tmp), str(product_tmp),
            placement, prod_scale,
            blur_bg, blur_strength,
            job_dir
        )
        update("✅ Product placed\n⏳ Detecting face...", 40)

        # Phase 3: Face
        face = detect_face(merged, job_dir)
        update("✅ Face detected\n⏳ Animating (30–120s on GPU)...", 55)

        # Phase 4: LivePortrait
        animated = run_liveportrait(
            face, audio_path, job_dir, motion_scale
        )
        update("✅ Animation done\n⏳ Rendering final video...", 80)

        # Phase 5: FFmpeg
        tw, th = ASPECT_RATIOS[aspect_key]
        grade  = COLOR_GRADES[color_key]
        grain_map = {"none": 0, "subtle": 12, "medium": 22, "strong": 35}
        grain  = grain_map.get(grain_level, 12)

        final = render_final(
            animated, audio_path, job_dir, job_id, duration,
            tw, th, grade,
            vignette, shake, shake_int, grain,
            fade_in, fade_out,
            caption_style, script, hook_text,
        )

        prog.progress(100)
        status.success(f"🎬 Done! ({time.time():.0f}s)")
        return final

    except Exception as e:
        status.error(f"❌ {type(e).__name__}: {e}")
        log.exception("Pipeline error")
        return None


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="UGC Ad Generator",
    page_icon="🎬",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stButton > button { border-radius: 8px; font-weight: 700; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("🎬 UGC Video Ad Generator")
st.markdown("**Realistic • Lip-synced • Cinematic** — face + product + script = ready-to-post ad")
st.divider()

left, right = st.columns([1, 1], gap="large")

# ═══════════════════════════════════════
# LEFT COLUMN — Inputs & Settings
# ═══════════════════════════════════════
with left:

    # ── Inputs ──────────────────────────
    st.subheader("📥 Inputs")
    human_file   = st.file_uploader(
        "👤 Human Image (front-facing, clear face)",
        type=["jpg", "jpeg", "png"])
    product_file = st.file_uploader(
        "📦 Product Image (PNG — transparent bg best)",
        type=["png"])
    script = st.text_area(
        "📝 Ad Script",
        height=130,
        placeholder=(
            "e.g. I've been using this for 30 days "
            "and my skin completely transformed. "
            "No harsh chemicals — just real results. "
            "Link in bio."
        ))

    st.divider()

    # ── Voice ───────────────────────────
    st.subheader("🎙️ Voice")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        voice_key = st.selectbox("Voice", list(TTS_VOICES.keys()))
    with col_v2:
        speed = st.select_slider(
            "Speed",
            options=["-20%", "-10%", "+0%", "+10%", "+20%"],
            value="+0%")

    st.divider()

    # ── Product ─────────────────────────
    st.subheader("📦 Product Placement")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        placement = st.selectbox("Position", PLACEMENTS)
    with col_p2:
        prod_scale = st.slider(
            "Size (% of width)", 0.10, 0.55, 0.28, 0.05)

    blur_bg = st.toggle("Portrait Mode Background Blur", value=False)
    blur_strength = 8
    if blur_bg:
        blur_strength = st.slider("Blur Strength", 2, 20, 8)

    st.divider()

    # ── Animation ───────────────────────
    st.subheader("🎭 Animation")
    motion_scale = st.slider(
        "Head Movement Intensity",
        min_value=0.3, max_value=2.0,
        value=1.0, step=0.1)

    st.divider()

    # ── Visual Effects ───────────────────
    st.subheader("🎨 Visual Effects")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        aspect_key = st.selectbox(
            "Aspect Ratio", list(ASPECT_RATIOS.keys()))
    with col_e2:
        color_key = st.selectbox(
            "Color Grade", list(COLOR_GRADES.keys()))

    col_e3, col_e4 = st.columns(2)
    with col_e3:
        grain_level = st.selectbox(
            "Film Grain",
            ["none", "subtle", "medium", "strong"],
            index=1)
    with col_e4:
        vignette = st.toggle("Vignette", value=True)

    shake     = st.toggle("Handheld Camera Shake", value=True)
    shake_int = 1.0
    if shake:
        shake_int = st.slider(
            "Shake Intensity", 0.3, 3.0, 1.0, 0.1)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        fade_in  = st.slider("Fade In (sec)",  0.0, 1.5, 0.4, 0.1)
    with col_f2:
        fade_out = st.slider("Fade Out (sec)", 0.0, 1.5, 0.5, 0.1)

    st.divider()

    # ── Captions ────────────────────────
    st.subheader("💬 Captions & Hook")
    caption_style = st.selectbox(
        "Caption Style",
        ["ugc", "subtitle", "none"],
        help="ugc = large bold (TikTok style), subtitle = smaller")
    hook_text = st.text_input(
        "Hook Text (shown first 2.5 seconds)",
        placeholder="Wait... this actually works?")

    st.divider()

    # ── Generate Button ──────────────────
    generate = st.button(
        "🎬 Generate UGC Ad",
        type="primary",
        use_container_width=True)

# ═══════════════════════════════════════
# RIGHT COLUMN — Output
# ═══════════════════════════════════════
with right:
    st.subheader("📤 Output")

    if generate:
        if not human_file:
            st.error("❌ Upload a human image."); st.stop()
        if not product_file:
            st.error("❌ Upload a product image."); st.stop()
        if not script.strip():
            st.error("❌ Enter an ad script."); st.stop()

        t0    = time.time()
        final = run_pipeline(
            human_file, product_file, script,
            voice_key, speed,
            placement, prod_scale,
            blur_bg, blur_strength,
            motion_scale,
            aspect_key, color_key,
            grain_level, vignette, shake, shake_int,
            fade_in, fade_out,
            caption_style, hook_text,
        )

        if final and final.exists():
            st.success(f"✅ Done in {time.time()-t0:.0f}s!")
            st.video(str(final))
            with open(str(final), "rb") as f:
                st.download_button(
                    label="⬇️ Download Video",
                    data=f,
                    file_name=f"ugc_ad_{final.stem}.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
    else:
        st.info(
            "👈 Fill in the inputs on the left\n\n"
            "**Steps:**\n"
            "1. Upload human image\n"
            "2. Upload product PNG\n"
            "3. Write your ad script\n"
            "4. Adjust settings\n"
            "5. Click Generate\n\n"
            "**Estimated time with GPU:**\n"
            "- Voice: 3–5s\n"
            "- Image: 2–3s\n"
            "- Animation: 30–90s\n"
            "- Render: 10–20s"
        )
