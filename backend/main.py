import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parent / "resnet18_mnist_baseline.pt"
IMAGE_SIZE = 28

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: nn.Module | None = None


def build_model() -> nn.Module:
    # Must mirror the architecture used at training time exactly: a timm
    # resnet18 with its first conv layer swapped for single-channel input.
    net = timm.create_model("resnet18", pretrained=False, num_classes=10)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    net.to(device)
    net.eval()
    return net


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = build_model()
    yield


app = FastAPI(title="MNIST Digit Recognizer API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: list[float]


def preprocess(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    pixels = np.array(image, dtype=np.float32)

    # Kaggle's Digit Recognizer pixels are a light stroke on a dark (0)
    # background, like MNIST. Raw uploads/drawings are usually the reverse
    # (dark stroke on a light background), so invert when that's detected.
    if pixels.mean() > 127:
        pixels = 255.0 - pixels

    pixels = pixels / 255.0
    tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    return tensor.to(device)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    try:
        tensor = preprocess(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        digit = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[digit].item())

    return PredictionResponse(
        digit=digit,
        confidence=confidence,
        probabilities=[float(p) for p in probabilities.tolist()],
    )
