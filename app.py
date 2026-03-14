import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request
import nltk

from image_generator import generate_image_from_prompt
from prompt_engine import build_visual_prompt, split_into_scenes

load_dotenv()

app = Flask(__name__)
app.config["IMAGES_DIR"] = Path("static") / "images"
app.config["MAX_SCENES"] = 8


def ensure_nltk_resources() -> None:
    """Ensure sentence tokenizer data exists."""
    for resource_path, resource_name in (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ):
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


@app.route("/", methods=["GET", "POST"])
def index():
    ensure_nltk_resources()

    storyboard = []
    error_message = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("story", "").strip()

        if not input_text:
            error_message = "Please enter a short paragraph before generating."
            return render_template(
                "index.html",
                storyboard=storyboard,
                error_message=error_message,
                input_text=input_text,
            )

        scenes = split_into_scenes(input_text)
        if not scenes:
            error_message = "No valid scenes were found in your text. Try writing 2-3 complete sentences."
            return render_template(
                "index.html",
                storyboard=storyboard,
                error_message=error_message,
                input_text=input_text,
            )

        if len(scenes) > app.config["MAX_SCENES"]:
            scenes = scenes[: app.config["MAX_SCENES"]]

        images_dir: Path = app.config["IMAGES_DIR"]
        images_dir.mkdir(parents=True, exist_ok=True)

        hf_api_key = os.getenv("HF_API_KEY", "").strip()
        if not hf_api_key:
            error_message = "HF_API_KEY is missing. Add it to your .env file and restart the app."
            return render_template(
                "index.html",
                storyboard=storyboard,
                error_message=error_message,
                input_text=input_text,
            )

        for index_num, scene in enumerate(scenes, start=1):
            prompt = build_visual_prompt(scene)
            image_filename = f"scene{index_num}.png"
            output_path = images_dir / image_filename

            image_result = generate_image_from_prompt(
                prompt=prompt,
                api_key=hf_api_key,
                output_path=output_path,
            )

            storyboard.append(
                {
                    "scene_number": index_num,
                    "caption": scene,
                    "prompt": prompt,
                    "image_path": f"images/{image_filename}",
                    "status": image_result["status"],
                    "error": image_result.get("error"),
                }
            )

        if all(item["status"] == "failed" for item in storyboard):
            error_message = "Image generation failed for all scenes. Check your API key or try again in a moment."

    return render_template(
        "index.html",
        storyboard=storyboard,
        error_message=error_message,
        input_text=input_text,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
