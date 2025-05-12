# handler.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import requests

app = FastAPI()

model_id = "naclbit/tranquil-garden-ghibli"  # Change if needed
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

class ImageRequest(BaseModel):
    image_url: str

@app.post("/")
async def generate_ghibli_image(payload: ImageRequest):
    try:
        image_url = payload.image_url
        if not image_url:
            raise HTTPException(status_code=400, detail="Missing image URL")

        prompt = "Studio Ghibli style reimagination of the provided image. Dreamlike, vivid, poetic."
        init_image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")
        init_image = init_image.resize((512, 512))

        result = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5)

        if not result or not result.images:
            raise HTTPException(status_code=500, detail="Failed to generate image")

        buffer = BytesIO()
        result.images[0].save(buffer, format="PNG")
        buffer.seek(0)

        return {"image": "data:image/png;base64," + buffer.getvalue().decode("latin1")}  # For internal test
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
