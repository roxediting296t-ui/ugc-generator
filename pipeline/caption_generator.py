"""
pipeline/caption_generator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generates word-by-word or phrase-by-phrase captions
synced to audio duration — burned into video via FFmpeg.

No Whisper needed — we already have the script + audio duration.
Splits script into timed segments and generates FFmpeg drawtext filters.

Caption styles:
  - ugc:      Large bold white, black outline, centered bottom (TikTok style)
  - subtitle: Smaller white, semi-transparent bg bar (YouTube style)
  - minimal:  Small white text, no background
"""

import logging
import re
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger("ugc.captions")


CAPTION_STYLES = {
    "ugc": {
        "fontsize": 72,
        "fontcolor": "white",
        "bordercolor": "black",
        "borderw": 4,
        "box": 0,
        "font": "Arial-Bold",
        "y_position": "h-h/6",    # bottom 1/6 of frame
        "words_per_segment": 3,
    },
    "subtitle": {
        "fontsize": 52,
        "fontcolor": "white",
        "bordercolor": "black",
        "borderw": 2,
        "box": 1,
        "boxcolor": "black@0.5",
        "font": "Arial",
        "y_position": "h-h/8",
        "words_per_segment": 6,
    },
    "minimal": {
        "fontsize": 44,
        "fontcolor": "white@0.9",
        "bordercolor": "black@0.6",
        "borderw": 2,
        "box": 0,
        "font": "Arial",
        "y_position": "h-h/10",
        "words_per_segment": 5,
    },
    "none": None,
}


def generate_caption_filter(
    script: str,
    audio_duration: float,
    style: str = "ugc",
    highlight_color: str = "yellow",
) -> str:
    """
    Generates an FFmpeg drawtext filter string for burned-in captions.

    Timing is estimated by distributing words proportionally across
    audio duration. Not frame-perfect, but good enough for UGC ads
    without needing Whisper transcription.

    Args:
        script:         Ad copy text.
        audio_duration: Total audio length in seconds.
        style:          One of: 'ugc', 'subtitle', 'minimal', 'none'.
        highlight_color: Color for the current word highlight (ugc style).

    Returns:
        FFmpeg vf filter string, or empty string if style='none'.
    """
    if style == "none" or style not in CAPTION_STYLES:
        return ""

    cfg = CAPTION_STYLES[style]
    if cfg is None:
        return ""

    segments = _split_into_segments(script, cfg["words_per_segment"])
    if not segments:
        return ""

    timed = _assign_timestamps(segments, audio_duration)
    filters = _build_drawtext_filters(timed, cfg, highlight_color)

    log.info(f"[CAPTION] Generated {len(timed)} caption segments ({style} style)")
    return ",".join(filters)


def _split_into_segments(script: str, words_per_segment: int) -> List[str]:
    """Split script into caption segments of N words each."""
    # Clean script — remove special chars that break FFmpeg drawtext
    clean = re.sub(r"[\"'\\:=\[\]{}@#$%^&*<>|]", "", script)
    clean = re.sub(r"\s+", " ", clean).strip()

    words = clean.split()
    segments = []

    for i in range(0, len(words), words_per_segment):
        chunk = " ".join(words[i:i + words_per_segment])
        segments.append(chunk)

    return segments


def _assign_timestamps(
    segments: List[str],
    total_duration: float,
) -> List[Tuple[str, float, float]]:
    """
    Assign start/end times to each segment proportionally.
    Longer segments get more time. Adds 0.1s padding at start/end.

    Returns:
        List of (text, start_time, end_time) tuples.
    """
    # Weight by character count (longer text = more screen time)
    weights = [len(s) for s in segments]
    total_weight = sum(weights)

    usable_duration = total_duration - 0.2  # 0.1s padding each side
    timed = []
    current_time = 0.1

    for i, (segment, weight) in enumerate(zip(segments, weights)):
        duration = (weight / total_weight) * usable_duration
        # Minimum 0.5s per segment, maximum 4s
        duration = max(0.5, min(4.0, duration))
        end_time = current_time + duration
        timed.append((segment, round(current_time, 3), round(end_time, 3)))
        current_time = end_time

    return timed


def _build_drawtext_filters(
    timed_segments: List[Tuple[str, float, float]],
    cfg: dict,
    highlight_color: str,
) -> List[str]:
    """Build one FFmpeg drawtext filter per caption segment."""
    filters = []

    for text, start, end in timed_segments:
        # Escape special chars for FFmpeg
        safe_text = (text
                     .replace("\\", "\\\\")
                     .replace("'", "\u2019")  # smart apostrophe
                     .replace(":", "\\:")
                     .replace("%", "\\%"))

        # Build drawtext filter
        parts = [
            f"text='{safe_text}'",
            f"fontsize={cfg['fontsize']}",
            f"fontcolor={cfg['fontcolor']}",
            f"bordercolor={cfg['bordercolor']}",
            f"borderw={cfg['borderw']}",
            f"x=(w-text_w)/2",        # horizontally centered
            f"y={cfg['y_position']}",
            f"enable='between(t,{start},{end})'",
        ]

        if cfg.get("box"):
            parts.append(f"box=1")
            parts.append(f"boxcolor={cfg.get('boxcolor', 'black@0.5')}")
            parts.append(f"boxborderw=12")

        # Font (FFmpeg uses system fonts — Arial is available on most systems)
        # On Ubuntu: sudo apt install fonts-liberation (provides Arial equivalent)
        # Fallback handled by FFmpeg automatically
        parts.append(f"font='{cfg['font']}'")

        filters.append(f"drawtext={':'.join(parts)}")

    return filters


def generate_hook_overlay(
    hook_text: str,
    duration: float = 2.5,
    style: str = "ugc",
) -> str:
    """
    Generates FFmpeg filter for an opening hook text overlay.
    Shown for the first `duration` seconds with a fade-in effect.

    Example hook_text: "This changed my skincare routine forever 👀"

    Args:
        hook_text: Short attention-grabbing opening text (1 line).
        duration:  How long hook stays on screen (seconds).
        style:     'ugc' or 'minimal'.

    Returns:
        FFmpeg filter string.
    """
    if not hook_text.strip():
        return ""

    safe = (hook_text
            .replace("\\", "\\\\")
            .replace("'", "\u2019")
            .replace(":", "\\:")
            .replace("%", "\\%"))

    # Emoji removal for FFmpeg compatibility (FFmpeg drawtext doesn't render emoji)
    safe = re.sub(r'[^\x00-\x7F]+', '', safe).strip()

    if not safe:
        return ""

    fontsize = 80 if style == "ugc" else 60
    fade_expr = f"if(lt(t,0.5),t/0.5,if(lt(t,{duration-0.5}),1,({duration}-t)/0.5))"

    filter_str = (
        f"drawtext="
        f"text='{safe}':"
        f"fontsize={fontsize}:"
        f"fontcolor=white:"
        f"bordercolor=black:"
        f"borderw=5:"
        f"font='Arial-Bold':"
        f"x=(w-text_w)/2:"
        f"y=h/4:"
        f"alpha='{fade_expr}':"
        f"enable='between(t,0,{duration})'"
    )

    return filter_str
