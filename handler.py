# handler.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import BytesIO
from PIL import Image, ImageOps
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import requests
import base64
import os
import random

app = FastAPI()

MODEL_ID = os.getenv("MYAZORA_MODEL_ID", "runwayml/stable-diffusion-v1-5")
LORA_PATH = os.getenv("MYAZORA_LORA_PATH")  # optional local path or repo id
LORA_SCALE_DEFAULT = float(os.getenv("MYAZORA_LORA_SCALE", "0.75"))

# create pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
).to("cuda")

# faster and cleaner sampler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# opt in perf helpers if available
if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
pipe.enable_attention_slicing()

# optional LoRA
if LORA_PATH:
    try:
        pipe.load_lora_weights(LORA_PATH)
    except Exception as e:
        print("LoRA load failed:", e)

class ImageRequest(BaseModel):
    image_url: str | None = None
    image_base64: str | None = None
    prompt: str | None = None
    negative_prompt: str | None = None
    strength: float | None = None
    guidance_scale: float | None = None
    steps: int | None = None
    seed: int | None = None
    lora_scale: float | None = None
    max_side: int | None = None  # resize while keeping aspect, default 768

def load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Could not fetch image, status {r.status_code}")
    img = Image.open(BytesIO(r.content))
    return img

def load_image_from_b64(data_url_or_b64: str) -> Image.Image:
    s = data_url_or_b64
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    b = base64.b64decode(s)
    img = Image.open(BytesIO(b))
    return img

def prepare_image(img: Image.Image, max_side: int = 768) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        if w >= h:
            new_w = max_side
            new_h = int(h * max_side / w)
        else:
            new_h = max_side
            new_w = int(w * max_side / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img

@app.post("/")
async def generate_image(req: ImageRequest):
    try:
        if not req.image_url and not req.image_base64:
            raise HTTPException(status_code=400, detail="Provide image_url or image_base64")

        # defaults tuned for identity preservation
        prompt = req.prompt or "Studio Ghibli inspired, soft colors, gentle lighting, clean line art, keep person identity and pose"
        negative = req.negative_prompt or "blurry, deformed, extra limbs, off model, heavy distortion, caricature, text, watermark"
        strength = max(0.05, min(0.6, req.strength if req.strength is not None else 0.28))
        guidance = req.guidance_scale if req.guidance_scale is not None else 4.5
        steps = max(10, min(60, req.steps if req.steps is not None else 28))
        seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
        lora_scale = req.lora_scale if req.lora_scale is not None else LORA_SCALE_DEFAULT
        max_side = req.max_side if req.max_side is not None else 768

        # init image
        if req.image_base64:
            init_img = load_image_from_b64(req.image_base64)
        else:
            init_img = load_image_from_url(req.image_url)

        init_img = prepare_image(init_img, max_side=max_side)

        # apply LoRA scale if loaded
        if LORA_PATH and hasattr(pipe, "fuse_lora"):
            pipe.fuse_lora(lora_scale=lora_scale)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        result = pipe(
            prompt=prompt,
            image=init_img,
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=steps,
            negative_prompt=negative,
            generator=generator
        )

        if LORA_PATH and hasattr(pipe, "unfuse_lora"):
            pipe.unfuse_lora()

        if not result or not result.images:
            raise HTTPException(status_code=500, detail="Image generation failed")

        out = result.images[0]
        buf = BytesIO()
        out.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return JSONResponse(content={"image": f"data:image/png;base64,{b64}", "seed": seed})

    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        print("Exception:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
