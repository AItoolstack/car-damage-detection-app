
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask
import shutil
import os
import uuid
from pathlib import Path
from typing import Optional
import json
import base64
from ultralytics import YOLO
import cv2
import numpy as np


# Templates directory
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "uploads")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "tiff", "tif"}

# Model paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DAMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "damage", "weights", "weights", "best.pt")
PARTS_MODEL_PATH = os.path.join(BASE_DIR, "models", "parts", "weights", "weights", "best.pt")

# Class names for parts
PARTS_CLASS_NAMES = ['headlamp', 'front_bumper', 'hood', 'door', 'rear_bumper']

# Helper: Run YOLO inference and return results
def run_yolo_inference(model_path, image_path, task='segment'):
    model = YOLO(model_path)
    results = model.predict(source=image_path, imgsz=640, conf=0.25, save=False, task=task)
    return results[0]

# Helper: Draw masks and confidence on image
def draw_masks_and_conf(image_path, yolo_result, class_names=None):
    img = cv2.imread(image_path)
    overlay = img.copy()
    out_img = img.copy()
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    for i, box in enumerate(yolo_result.boxes):
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        color = colors[cls % len(colors)]
        # Draw bbox
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls] if class_names else 'damage'}: {conf:.2f}"
        cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Draw mask if available
        if hasattr(yolo_result, 'masks') and yolo_result.masks is not None:
            mask = yolo_result.masks.data[i].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (x2-x1, y2-y1))
            roi = overlay[y1:y2, x1:x2]
            colored_mask = np.zeros_like(roi)
            colored_mask[mask > 127] = color
            overlay[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.5, colored_mask, 0.5, 0)
    out_img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    return out_img

# Helper: Generate JSON output
def generate_json_output(filename, damage_result, parts_result):
    # Damage severity: use max confidence
    severity_score = float(max([float(box.conf[0]) for box in damage_result.boxes], default=0))
    damage_regions = []
    for box in damage_result.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        damage_regions.append({"bbox": [x1, y1, x2, y2], "confidence": conf})
    # Parts
    parts = []
    for i, box in enumerate(parts_result.boxes):
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        # Damage %: use mask area / bbox area if available
        damage_percentage = None
        if hasattr(parts_result, 'masks') and parts_result.masks is not None:
            mask = parts_result.masks.data[i].cpu().numpy()
            mask_area = np.sum(mask > 0.5)
            bbox_area = (x2-x1)*(y2-y1)
            damage_percentage = float(mask_area / bbox_area) if bbox_area > 0 else None
        parts.append({
            "part": PARTS_CLASS_NAMES[cls] if cls < len(PARTS_CLASS_NAMES) else str(cls),
            "damaged": True,
            "confidence": conf,
            "damage_percentage": damage_percentage,
            "bbox": [x1, y1, x2, y2]
        })
    # Optionally, add base64 masks
    # (not implemented here for brevity)
    return {
        "filename": filename,
        "damage": {
            "severity_score": severity_score,
            "regions": damage_regions
        },
        "parts": parts,
        "cost_estimate": None
    }

# Dummy login credentials
def check_login(username: str, password: str) -> bool:
    return username == "demo" and password == "demo123"

@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@router.post("/login", response_class=HTMLResponse)
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if check_login(username, password):
        return templates.TemplateResponse("index.html", {"request": request, "result": None, "user": username})
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/upload", response_class=HTMLResponse)
def upload_image(request: Request, file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Unsupported file type."})

    # Save uploaded file
    session_id = str(uuid.uuid4())
    upload_path = os.path.join(UPLOAD_DIR, f"{session_id}.{ext}")
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run both inferences
    try:
        damage_result = run_yolo_inference(DAMAGE_MODEL_PATH, upload_path)
        parts_result = run_yolo_inference(PARTS_MODEL_PATH, upload_path)

        # Save annotated images
        damage_img_path = os.path.join(RESULTS_DIR, f"{session_id}_damage.png")
        parts_img_path = os.path.join(RESULTS_DIR, f"{session_id}_parts.png")
        json_path = os.path.join(RESULTS_DIR, f"{session_id}_result.json")
        damage_img_url = f"/static/results/{session_id}_damage.png"
        parts_img_url = f"/static/results/{session_id}_parts.png"
        json_url = f"/static/results/{session_id}_result.json"

        # Defensive: set to None by default
        damage_img = None
        parts_img = None
        json_output = None

        # Only save and set if inference returns boxes
        if hasattr(damage_result, 'boxes') and len(damage_result.boxes) > 0:
            damage_img = draw_masks_and_conf(upload_path, damage_result)
            cv2.imwrite(damage_img_path, damage_img)
        if hasattr(parts_result, 'boxes') and len(parts_result.boxes) > 0:
            parts_img = draw_masks_and_conf(upload_path, parts_result, class_names=PARTS_CLASS_NAMES)
            cv2.imwrite(parts_img_path, parts_img)
        if (hasattr(damage_result, 'boxes') and len(damage_result.boxes) > 0) or (hasattr(parts_result, 'boxes') and len(parts_result.boxes) > 0):
            json_output = generate_json_output(file.filename, damage_result, parts_result)
            with open(json_path, "w") as jf:
                json.dump(json_output, jf, indent=2)

        # Prepare URLs for download (only if files exist)
        result = {
            "filename": file.filename,
            "damage_image": damage_img_url if damage_img is not None else None,
            "parts_image": parts_img_url if parts_img is not None else None,
            "json": json_output,
            "json_download": json_url if json_output is not None else None
        }
        # Debug log
        print("[DEBUG] Result dict:", result)
    except Exception as e:
        result = {
            "filename": file.filename,
            "error": f"Inference failed: {str(e)}",
            "damage_image": None,
            "parts_image": None,
            "json": None,
            "json_download": None
        }
        print("[ERROR] Inference failed:", e)

    import threading
    import time
    def delayed_cleanup():
        time.sleep(300)  # 5 minutes
        try:
            os.remove(upload_path)
        except Exception:
            pass
        for suffix in ["_damage.png", "_parts.png", "_result.json"]:
            try:
                os.remove(os.path.join(RESULTS_DIR, f"{session_id}{suffix}"))
            except Exception:
                pass

    threading.Thread(target=delayed_cleanup, daemon=True).start()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "original_image": f"/static/uploads/{session_id}.{ext}"
        }
    )
