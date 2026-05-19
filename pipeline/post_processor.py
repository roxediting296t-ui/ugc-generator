"""
pipeline/post_processor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Final video assembly with all UGC-quality effects:

  ✅ Resolution scaling (9:16 / 1:1 / 16:9)
  ✅ Color grading presets (warm, cool, neutral, cinematic)
  ✅ Film grain (makes it look shot on phone)
  ✅ Vignette (natural lens darkening at edges)
  ✅ Handheld camera shake simulation
  ✅ Burned-in captions
  ✅ Hook text overlay
  ✅ Background music mixing
  ✅ Smooth intro/outro fade
  ✅ H.264 High CRF 16 output (near-lossless)
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger("ugc.post")


# ── Resolution presets ───────────────────────────────────────────────────────
ASPECT_RATIOS = {
    "9:16 Portrait (TikTok/Reels)": (1080, 1920),
    "1:1 Square (Instagram Feed)":  (1080, 1080),
    "16:9 Landscape (YouTube)":     (1920, 1080),
    "4:5 Portrait (Instagram)":     (1080, 1350),
}

# ── Color grading presets (FFmpeg eq + curves filters) ───────────────────────
COLOR_GRADES = {
    "warm":      "eq=brightness=0.03:contrast=1.10:saturation=1.15:gamma_r=1.08:gamma_b=0.94",
    "cool":      "eq=brightness=0.02:contrast=1.08:saturation=1.10:gamma_r=0.94:gamma_b=1.08",
    "neutral":   "eq=brightness=0.01:contrast=1.05:saturation=1.05",
    "cinematic": "eq=brightness=-0.02:contrast=1.15:saturation=0.90:gamma=1.05",
    "vibrant":   "eq=brightness=0.04:contrast=1.12:saturation=1.35",
    "matte":     "eq=brightness=0.05:contrast=0.95:saturation=0.85:gamma=1.10",
}

# ── Film grain intensities ────────────────────────────────────────────────────
GRAIN_LEVELS = {
    "none":   0,
    "subtle": 12,
    "medium": 22,
    "strong": 35,
}


def post_process(
    animated_video: Path,
    audio_path: Path,
    job_dir: Path,
    output_path: Path,
    # Layout
    aspect_ratio: str = "9:16 Portrait (TikTok/Reels)",
    # Color
    color_grade: str = "warm",
    # Effects
    grain_level: str = "subtle",
    enable_vignette: bool = True,
    enable_camera_shake: bool = True,
    shake_intensity: float = 1.0,
    # Fade
    fade_in_duration: float = 0.4,
    fade_out_duration: float = 0.5,
    # Captions
    caption_filter: str = "",
    hook_filter: str = "",
    # Audio
    bgm_path: Optional[Path] = None,
    bgm_volume: float = 0.12,
    voice_volume: float = 1.0,
) -> Path:
    """
    Final assembly. Builds one FFmpeg command with all effects chained.

    Args:
        animated_video:     LivePortrait output .mp4.
        audio_path:         TTS voice .mp3.
        job_dir:            Job directory.
        output_path:        Where to save final video.
        aspect_ratio:       From ASPECT_RATIOS keys.
        color_grade:        From COLOR_GRADES keys.
        grain_level:        From GRAIN_LEVELS keys ('none'/'subtle'/'medium'/'strong').
        enable_vignette:    Dark edges for natural lens look.
        enable_camera_shake: Subtle handheld motion.
        shake_intensity:    0.5=very subtle, 1.0=normal, 2.0=shaky.
        fade_in_duration:   Seconds for black fade-in at start.
        fade_out_duration:  Seconds for black fade-out at end.
        caption_filter:     FFmpeg drawtext filter string from caption_generator.
        hook_filter:        FFmpeg drawtext filter for hook text.
        bgm_path:           Optional background music file.
        bgm_volume:         BGM volume (0.0–1.0). Keep 0.08–0.15 for UGC.
        voice_volume:       Voice volume multiplier.

    Returns:
        Path to final output video.
    """
    target_w, target_h = ASPECT_RATIOS.get(
        aspect_ratio, ASPECT_RATIOS["9:16 Portrait (TikTok/Reels)"]
    )

    log.info(f"[POST] Assembling final video: {target_w}×{target_h}")
    log.info(f"[POST] Color: {color_grade} | Grain: {grain_level} | "
             f"Vignette: {enable_vignette} | Shake: {enable_camera_shake}")

    # ── Build video filter chain ─────────────────────────────────────────────
    vf_chain = []

    # 1. Scale and pad to target resolution
    vf_chain.append(
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1"
    )

    # 2. Color grade
    grade = COLOR_GRADES.get(color_grade, COLOR_GRADES["warm"])
    vf_chain.append(grade)

    # 3. Handheld camera shake
    # Uses FFmpeg geq (pixel expression) to apply sinusoidal offset
    # Simulates natural micro-movement of hand-held phone recording
    if enable_camera_shake:
        si = shake_intensity
        shake = (
            f"crop="
            f"w={target_w}:h={target_h}:"
            f"x='({target_w}*0.01*{si})*sin(t*2.1+0.3)':"
            f"y='({target_h}*0.008*{si})*sin(t*1.7+1.1)',"
            f"scale={target_w}:{target_h}"
        )
        vf_chain.append(shake)

    # 4. Film grain
    grain_strength = GRAIN_LEVELS.get(grain_level, 12)
    if grain_strength > 0:
        # Uses noise filter — simulates phone camera sensor noise
        grain = f"noise=alls={grain_strength}:allf=t+u"
        vf_chain.append(grain)

    # 5. Vignette
    if enable_vignette:
        # Creates natural lens darkening at edges
        vignette = f"vignette=angle=PI/4:mode=forward"
        vf_chain.append(vignette)

    # 6. Fade in / out
    # We'll add fade after we know the duration
    # fade filters added below after duration check

    # 7. Caption overlay
    if caption_filter:
        vf_chain.append(caption_filter)

    # 8. Hook text overlay
    if hook_filter:
        vf_chain.append(hook_filter)

    vf_string = ",".join(vf_chain)

    # ── Build audio filter chain ─────────────────────────────────────────────
    # We need to know video duration for fade-out timing
    # Use a two-pass approach: first get duration, then build full command

    duration = _get_video_duration(animated_video)
    if duration is None:
        duration = 15.0  # safe fallback

    # Add fade in/out to video filter
    fade_filters = []
    if fade_in_duration > 0:
        fade_filters.append(
            f"fade=t=in:st=0:d={fade_in_duration}:color=black"
        )
    if fade_out_duration > 0:
        fade_start = max(0, duration - fade_out_duration)
        fade_filters.append(
            f"fade=t=out:st={fade_start:.3f}:d={fade_out_duration}:color=black"
        )

    if fade_filters:
        vf_string = vf_string + "," + ",".join(fade_filters)

    # ── Build FFmpeg command ──────────────────────────────────────────────────
    cmd = ["ffmpeg", "-y"]

    # Inputs
    cmd.extend(["-i", str(animated_video)])   # input 0: video
    cmd.extend(["-i", str(audio_path)])        # input 1: voice

    has_bgm = bgm_path and Path(bgm_path).exists()
    if has_bgm:
        cmd.extend(["-i", str(bgm_path)])      # input 2: background music
        cmd.extend(["-stream_loop", "-1"])     # loop BGM if shorter than video

    # Video filter
    cmd.extend(["-vf", vf_string])

    # Audio mixing
    if has_bgm:
        # Mix voice + BGM with volume controls
        audio_filter = (
            f"[1:a]volume={voice_volume}[voice];"
            f"[2:a]volume={bgm_volume},afade=t=out:st={max(0,duration-1.5):.2f}:d=1.5[bgm];"
            f"[voice][bgm]amix=inputs=2:duration=first:dropout_transition=2[audio_out]"
        )
        cmd.extend(["-filter_complex", audio_filter])
        cmd.extend(["-map", "0:v:0"])
        cmd.extend(["-map", "[audio_out]"])
    else:
        # Voice only with volume control
        cmd.extend(["-map", "0:v:0"])
        cmd.extend(["-map", "1:a:0"])
        if voice_volume != 1.0:
            cmd.extend(["-af", f"volume={voice_volume}"])

    # Audio fade out
    if fade_out_duration > 0 and not has_bgm:
        af_fade = f"afade=t=out:st={max(0, duration - fade_out_duration):.3f}:d={fade_out_duration}"
        cmd.extend(["-af", af_fade])

    # Encoding settings
    cmd.extend([
        "-c:v", "libx264",
        "-crf", "16",           # near-lossless (18=visually lossless, 16=higher quality)
        "-preset", "slow",      # better compression, worth the encode time
        "-profile:v", "high",
        "-level", "4.1",
        "-pix_fmt", "yuv420p",  # broadest compatibility
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-shortest",
        "-movflags", "+faststart",  # web-optimized (moov atom at start)
        str(output_path),
    ])

    log.info(f"[POST] FFmpeg command built ({len(cmd)} args)")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        log.error("[POST] FFmpeg stderr:\n" + result.stderr[-3000:])
        raise RuntimeError(
            f"FFmpeg failed (exit {result.returncode}).\n"
            f"STDERR (last 2000 chars):\n{result.stderr[-2000:]}"
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info(f"[POST] Final video: {output_path} ({size_mb:.1f} MB, {duration:.1f}s)")
    return output_path


def upscale_with_realesrgan(
    input_video: Path,
    job_dir: Path,
    weights_path: str,
) -> Path:
    """
    Frame-by-frame Real-ESRGAN x4 upscaling.
    Converts 512×512 LivePortrait output to ~2048×2048 before scaling to 1080p.
    Result: sharp, detailed faces instead of blurry bicubic upscale.

    Requires: pip install realesrgan basicsr
    Weights:  RealESRGAN_x4plus.pth (face-optimized)
    """
    try:
        import cv2
        import numpy as np
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        log.warning(
            "[SR] realesrgan/basicsr not installed — skipping upscale.\n"
            "Install: pip install realesrgan basicsr"
        )
        return input_video

    if not Path(weights_path).exists():
        log.warning(f"[SR] Weights not found: {weights_path} — skipping upscale.")
        return input_video

    log.info("[SR] Starting Real-ESRGAN upscaling...")

    frames_dir = job_dir / "frames_raw"
    upscaled_dir = job_dir / "frames_up"
    frames_dir.mkdir(exist_ok=True)
    upscaled_dir.mkdir(exist_ok=True)

    # Extract frames
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_video),
         "-q:v", "1", str(frames_dir / "frame_%05d.png")],
        check=True, capture_output=True
    )

    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise RuntimeError("[SR] No frames extracted.")

    log.info(f"[SR] Upscaling {len(frames)} frames...")

    model = RRDBNet(num_in_ch=3, num_out_ch=3,
                    num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=weights_path,
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True,
    )

    for frame in frames:
        img = cv2.imread(str(frame), cv2.IMREAD_UNCHANGED)
        upscaled, _ = upsampler.enhance(img, outscale=4)
        cv2.imwrite(str(upscaled_dir / frame.name), upscaled)

    # Rebuild video from upscaled frames
    upscaled_video = job_dir / "animated_upscaled.mp4"
    fps = _get_video_fps(input_video)

    subprocess.run(
        ["ffmpeg", "-y",
         "-framerate", str(fps),
         "-i", str(upscaled_dir / "frame_%05d.png"),
         "-c:v", "libx264", "-crf", "16", "-preset", "slow",
         "-pix_fmt", "yuv420p", str(upscaled_video)],
        check=True, capture_output=True
    )

    log.info(f"[SR] Upscaling complete: {upscaled_video}")
    return upscaled_video


def _get_video_duration(video_path: Path) -> Optional[float]:
    """Get video duration in seconds via ffprobe."""
    import json
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(video_path)],
            capture_output=True, text=True, timeout=15
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return None


def _get_video_fps(video_path: Path) -> float:
    """Get video FPS via ffprobe."""
    import json
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(video_path)],
            capture_output=True, text=True, timeout=15
        )
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                r = stream.get("r_frame_rate", "25/1")
                num, den = r.split("/")
                return float(num) / float(den)
    except Exception:
        pass
    return 25.0
