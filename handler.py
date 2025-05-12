# handler.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import base64
import requests

app = FastAPI()

pipe = None  # Global placeholder for the model

class ImageRequest(BaseModel):
    image_url: str

@app.on_event("startup")
async def load_model():
    global pipe
    model_id = "naclbit/tranquil-garden-ghibli"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda")

@app.post("/")
async def generate_ghibli_image(payload: ImageRequest):
    try:
        if not payload.image_url:
            raise HTTPException(status_code=400, detail="Missing image URL")

        prompt = "Studio Ghibli style reimagination of the provided image. Dreamlike, vivid, poetic."
        response = requests.get(payload.image_url)
        response.raise_for_status()

        init_image = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))

        result = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5)

        if not result.images:
            raise HTTPException(status_code=500, detail="Image generation failed")

        buffer = BytesIO()
        result.images[0].save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"image": f"data:image/png;base64,{base64_image}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
