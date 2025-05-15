from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import requests
import base64

app = FastAPI()

# Load the model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

# Request payload schema
class ImageRequest(BaseModel):
    image_url: str

# Main API endpoint
@app.post("/")
async def generate_image(request: ImageRequest):
    try:
        # Load and preprocess image
        response = requests.get(request.image_url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((512, 512))

        # Ghibli-style prompt
        prompt = "Studio Ghibli style reimagination of the provided image. Dreamlike, vivid, poetic."

        result = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5)

        if not result or not result.images:
            raise HTTPException(status_code=500, detail="Image generation failed")

        # Encode to base64
        buffer = BytesIO()
        result.images[0].save(buffer, format="PNG")
        base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"imageBase64": f"data:image/png;base64,{base64_img}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server only if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("handler:app", host="0.0.0.0", port=3000)
