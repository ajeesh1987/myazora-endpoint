# handler.py
import os
import torch
from diffusers import StableDiffusionPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO
import uuid

app = FastAPI()

class Input(BaseModel):
    imageUrl: str

model = None

def load_pipeline():
    global model
    model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    model.to("cuda")

@app.on_event("startup")
async def startup_event():
    load_pipeline()

@app.post("/")
async def generate_image(input: Input):
    try:
        response = requests.get(input.imageUrl)
        response.raise_for_status()
        init_image = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))
        prompt = "Studio Ghibli style reinterpretation of the uploaded image. Soft colors, nature background, dreamy lighting"
        image = model(prompt=prompt, image=init_image).images[0]

        temp_path = f"/tmp/{uuid.uuid4().hex}.png"
        image.save(temp_path)

        return {"image_path": temp_path}  # This will be adapted by RunPod's return policy

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
