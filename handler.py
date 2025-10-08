from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import requests
import base64
app = FastAPI()

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

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

        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return JSONResponse(content={"image": f"data:image/png;base64,{image_base64}"})
    except Exception as e:
        print("❌ Exception:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
