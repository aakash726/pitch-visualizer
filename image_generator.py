import time
from pathlib import Path
from typing import Dict

import requests

HF_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
HF_MODEL_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}"


def generate_image_from_prompt(prompt: str, api_key: str, output_path: Path) -> Dict[str, str]:
    """Generate an image from text using Hugging Face Inference API."""
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True},
    }

    try:
        response = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=120)

        if response.status_code == 503:
            # Brief retry path in case the model is loading.
            time.sleep(3)
            response = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            error_text = response.text[:300]
            if response.status_code == 403 and "Inference Providers" in response.text:
                error_text = (
                    "Your HF token is missing Inference Providers permission. "
                    "Create a fine-grained token with 'Make calls to Inference Providers' and update HF_API_KEY."
                )
            elif response.status_code == 410 and "api-inference.huggingface.co" in response.text:
                error_text = (
                    "Deprecated endpoint detected. Router endpoint is now required by Hugging Face."
                )

            return {
                "status": "failed",
                "error": f"HTTP {response.status_code}: {error_text}",
            }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)

        return {"status": "success"}

    except requests.RequestException as exc:
        return {
            "status": "failed",
            "error": f"Request error: {exc}",
        }
