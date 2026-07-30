import base64
import io
import os
import time

import requests
import runpod
import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image

MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
HF_TOKEN = os.getenv("HF_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = AutoPipelineForImage2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    use_safetensors=True,
    token=HF_TOKEN,
)
pipe = pipe.to(device)


def load_image(image_url):
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def handler(job):
    started = time.time()
    data = job.get("input", {})

    image_url = data["image_url"]
    prompt = data.get(
        "prompt",
        "commercial dealership vehicle photo, realistic lighting, natural shadows",
    )
    negative_prompt = data.get(
        "negative_prompt",
        "distorted car, changed vehicle shape, bad wheels, blurry, fake, cartoon",
    )

    strength = float(data.get("strength", 0.22))
    steps = int(data.get("steps", 25))
    guidance = float(data.get("guidance_scale", 5.0))

    image = load_image(image_url)
    image = image.resize((1024, 768))

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance,
        num_inference_steps=steps,
    ).images[0]

    return {
        "status": "completed",
        "model": MODEL_ID,
        "runtime_ms": round((time.time() - started) * 1000),
        "image_base64": image_to_base64(result),
    }


runpod.serverless.start({"handler": handler})