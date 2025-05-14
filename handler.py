from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import requests

app = FastAPI()

model_id = "naclbit/tranquil-garden-ghibli"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

class ImageRequest(BaseModel):
    image_url: str

@app.post("/")
async def generate_image(request: ImageRequest):
    try:
        response = requests.get(request.image_url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((512, 512))

        prompt = "Studio Ghibli style reimagination of the provided image. Dreamlike, vivid, poetic."
        result = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5)

        if not result or not result.images:
            raise HTTPException(status_code=500, detail="Image generation failed")

        buffer = BytesIO()
        result.images[0].save(buffer, format="PNG")
        buffer.seek(0)

        return {"image": "data:image/png;base64," + buffer.getvalue().decode("latin1")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
