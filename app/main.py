from typing import Union
import io

import torch
from fastapi import FastAPI, UploadFile, File, Response
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid

from app.bigram_model import BigramModel
from app.embedding_model import calculate_embedding
from helper_lib.model import get_model
from helper_lib.utils import load_model

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
cnn_model = get_model("CNN")
cnn_model = load_model(cnn_model, "cnn_model.pth", device="cpu")
cnn_model.eval()

gan_generator, _ = get_model("GAN")
gan_generator = load_model(gan_generator, "generator.pth", device="cpu")
gan_generator.eval()

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = calculate_embedding(request.query)
    return {"query": request.query, "embedding": embedding}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32))

    transform = transforms.ToTensor()
    tensor = transform(image).unsqueeze(0)

    output = cnn_model(tensor)
    _, predicted = torch.max(output, 1)
    predicted_class = CLASSES[predicted.item()]

    return {"predicted_class": predicted_class}

@app.post("/generate_digit")
def generate_digit(num_samples: int = 10):
    noise = torch.randn(num_samples, 100)
    with torch.no_grad():
        images = gan_generator(noise)

    grid = make_grid(images, nrow=5, normalize=True)
    ndarr = grid.mul(255).byte().permute(1, 2, 0).numpy()
    img = Image.fromarray(ndarr)

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return Response(content=buf.getvalue(), media_type="image/png")
