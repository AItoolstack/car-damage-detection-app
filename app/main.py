
# --- Model download logic (Hugging Face Hub) ---
import os
import requests

def download_if_missing(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading model from {url} to {dest}...")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")

# Hugging Face direct download links
DAMAGE_MODEL_URL = "https://huggingface.co/AItoolstack/car_damage_detection/resolve/main/yolov8_models/damage/weights/weights/best.pt"
PARTS_MODEL_URL = "https://huggingface.co/AItoolstack/car_damage_detection/resolve/main/yolov8_models/parts/weights/weights/best.pt"

DAMAGE_MODEL_PATH = os.path.join("models", "damage", "weights", "weights", "best.pt")
PARTS_MODEL_PATH = os.path.join("models", "parts", "weights", "weights", "best.pt")

download_if_missing(DAMAGE_MODEL_URL, DAMAGE_MODEL_PATH)
download_if_missing(PARTS_MODEL_URL, PARTS_MODEL_PATH)

from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import aiofiles
from pathlib import Path
import uuid
from app.routers import inference

app = FastAPI(title="Car Damage Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(inference.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
