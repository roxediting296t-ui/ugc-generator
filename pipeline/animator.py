"""
pipeline/animator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LivePortrait integration for lip-sync + natural head movement.
"""

import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("ugc.animator")


def animate_video(
    face_image_path: Path,
    audio_path: Path,
    job_dir: Path,
    live_portrait_dir: Path,
    motion_scale: float = 1.0,
    enable_eye_retargeting: bool = True,
    enable_lip_retargeting: bool = True,
) -> Path:
    """
    Runs LivePortrait inference to produce lip-synced animated video.

    Args:
        face_image_path:        Preprocessed 512×512 face PNG.
        audio_path:             TTS audio .mp3.
        job_dir:                Job working directory.
        live_portrait_dir:      Path to cloned LivePortrait repo.
        motion_scale:           Head movement intensity (0.5=subtle, 1.5=expressive).
        enable_eye_retargeting: Natural eye blinks.
        enable_lip_retargeting: Audio-driven lip sync.

    Returns:
        Path to animated .mp4 video.
    """
    script = live_portrait_dir / "inference.py"

    if not live_portrait_dir.exists():
        raise FileNotFoundError(
            f"LivePortrait not found: {live_portrait_dir}\n"
            "Fix: git clone https://github.com/KwaiVGI/LivePortrait.git\n"
            "     export LIVE_PORTRAIT_DIR=/path/to/LivePortrait"
        )

    if not script.exists():
        raise FileNotFoundError(
            f"inference.py not found in {live_portrait_dir}\n"
            "Check your LivePortrait version — entry point may differ."
        )

    output_video = job_dir / "animated_raw.mp4"

    cmd = [
        sys.executable, str(script),
        "--source_image",    str(face_image_path),
        "--audio",           str(audio_path),
        "--output_dir",      str(job_dir),
        "--output_name",     "animated_raw",
        "--flag_relative",
        "--flag_pasteback",
        "--flag_do_crop",
    ]

    if enable_eye_retargeting:
        cmd.append("--flag_eye_retargeting")
    if enable_lip_retargeting:
        cmd.append("--flag_lip_retargeting")
    if motion_scale != 1.0:
        cmd.extend(["--driving_multiplier", str(motion_scale)])

    log.info(f"[ANIM] Running LivePortrait...")
    log.info(f"[ANIM] Source: {face_image_path}")
    log.info(f"[ANIM] Audio:  {audio_path}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(live_portrait_dir),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "[ANIM] LivePortrait timed out (10 min). "
            "Causes: no GPU, very long audio, or hung process."
        )

    if result.returncode != 0:
        log.error("[ANIM] STDERR:\n" + result.stderr[-2000:])
        raise RuntimeError(
            f"LivePortrait failed (exit {result.returncode}).\n\n"
            f"Common causes:\n"
            f"  • Missing pretrained_weights/ folder\n"
            f"  • Wrong Python environment (activate LivePortrait venv)\n"
            f"  • CUDA out of memory (reduce image size)\n\n"
            f"STDERR:\n{result.stderr[-1500:]}"
        )

    # Find output — LivePortrait may name it differently
    if not output_video.exists():
        candidates = sorted(job_dir.glob("*.mp4"))
        if candidates:
            output_video = candidates[0]
            log.warning(f"[ANIM] Using alternate output: {output_video.name}")
        else:
            raise FileNotFoundError(
                "[ANIM] LivePortrait produced no .mp4 output.\n"
                f"STDOUT:\n{result.stdout[-1000:]}"
            )

    log.info(f"[ANIM] Animation complete: {output_video}")
    return output_video
