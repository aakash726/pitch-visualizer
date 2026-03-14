from typing import List

import nltk

VISUAL_STYLE_SUFFIX = (
    "cinematic composition, expressive lighting, dynamic perspective, "
    "highly detailed digital illustration"
)


def split_into_scenes(text: str) -> List[str]:
    """Split user paragraph into cleaned sentence-level scenes."""
    scenes = [sentence.strip() for sentence in nltk.sent_tokenize(text) if sentence.strip()]
    return scenes


def build_visual_prompt(scene_sentence: str) -> str:
    """Convert a plain sentence into a richer visual prompt for image generation."""
    lower_scene = scene_sentence.lower()

    if "startup" in lower_scene or "office" in lower_scene:
        context = "modern startup office environment"
    elif "customer" in lower_scene or "support" in lower_scene:
        context = "customer service atmosphere with screens and chat interfaces"
    elif "growth" in lower_scene or "doubled" in lower_scene or "success" in lower_scene:
        context = "uplifting business success moment with confident team"
    else:
        context = "story-driven scene with clear subject and setting"

    return f"{scene_sentence}, {context}, {VISUAL_STYLE_SUFFIX}"
