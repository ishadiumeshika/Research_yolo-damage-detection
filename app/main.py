from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import certifi
import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from ultralytics import YOLO


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB = os.getenv("MONGODB_DB", "packaging_review")
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "").strip()
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "120"))
MAX_SHIPPING_WEIGHT_KG = float(os.getenv("MAX_SHIPPING_WEIGHT_KG", "30"))
FRAME_SAMPLE_COUNT = int(os.getenv("FRAME_SAMPLE_COUNT", "24"))
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", "0.25"))

LABEL_TO_RULE = {
    "export_mark_label": "shipping_label",
    "fragile_label": "fragile",
    "handle_with_care_label": "handle_with_care",
    "keep_dry_label": "keep_dry",
    "protect_from_heat_label": "protect_from_heat",
    "this_side_up_label": "this_side_up",
    "package_damage": "damage",
}

KNOWN_DECLARED_TERMS = {
    "carton box": "carton_box",
    "carton": "carton_box",
    "wooden box": "wooden_box",
    "wooden": "wooden_box",
    "shipping label": "export_mark_label",
    "label": "export_mark_label",
    "fragile": "fragile_label",
    "keep dry": "keep_dry_label",
    "handle with care": "handle_with_care_label",
    "protect from heat": "protect_from_heat_label",
    "this side up": "this_side_up_label",
    "damage": "package_damage",
}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def parse_bool(value: Optional[str]) -> bool:
    return normalize_text(value or "") in {"1", "true", "on", "yes", "y"}


def split_declared_items(value: str) -> list[str]:
    parts = re.split(r"[,;\n]+", value)
    return [normalize_text(part) for part in parts if normalize_text(part)]


def build_mongo_client() -> Optional[MongoClient]:
    if not MONGODB_URI:
        return None
    return MongoClient(MONGODB_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)


def load_yolo_model() -> Optional[YOLO]:
    candidate_paths = []
    if MODEL_WEIGHTS:
        candidate_paths.append(Path(MODEL_WEIGHTS))
    candidate_paths.extend(
        [
            BASE_DIR / "best.pt",
            BASE_DIR / "models" / "best.pt",
            Path("best.pt"),
            Path("models/best.pt"),
        ]
    )

    for candidate in candidate_paths:
        if candidate.exists():
            return YOLO(str(candidate))
    return None


def sample_frame_indexes(frame_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    sample_count = max(1, min(FRAME_SAMPLE_COUNT, frame_count))
    if sample_count == 1:
        return [0]
    step = max(1, frame_count // sample_count)
    indexes = list(range(0, frame_count, step))[:sample_count]
    if indexes[-1] != frame_count - 1:
        indexes.append(frame_count - 1)
    return sorted(set(indexes))


def analyze_video(video_path: Path, model: Optional[YOLO]) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="The uploaded video could not be read.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0
    if duration and duration > MAX_VIDEO_SECONDS:
        raise HTTPException(status_code=400, detail=f"Video must be {MAX_VIDEO_SECONDS} seconds or shorter.")

    detections: dict[str, float] = {}
    if model is not None and frame_count > 0:
        for index in sample_frame_indexes(frame_count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            results = model.predict(frame, conf=DETECTION_CONFIDENCE, verbose=False)
            if not results:
                continue

            result = results[0]
            names = getattr(result, "names", None) or getattr(model, "names", {})
            for box in getattr(result, "boxes", []):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = names.get(class_id, str(class_id))
                detections[class_name] = max(detections.get(class_name, 0.0), confidence)

    capture.release()
    return {"duration_seconds": round(duration, 2), "detections": detections}


def classify_submission(
    declared_contents: list[str],
    declared_flags: dict[str, bool],
    detections: dict[str, float],
    weight_kg: float,
) -> dict[str, Any]:
    detected_names = sorted(detections.keys())
    detected_rules = {LABEL_TO_RULE.get(name, name) for name in detected_names}

    warnings: list[str] = []
    blockers: list[str] = []
    missing_declared_items: list[str] = []

    if weight_kg > MAX_SHIPPING_WEIGHT_KG:
        blockers.append(
            f"Declared weight {weight_kg:.2f} kg exceeds the shipping limit of {MAX_SHIPPING_WEIGHT_KG:.2f} kg."
        )

    if detections.get("package_damage", 0.0) >= 0.25:
        blockers.append("Damage was detected on the package.")

    for declared_item in declared_contents:
        matched_class = None
        for key, class_name in KNOWN_DECLARED_TERMS.items():
            if key in declared_item:
                matched_class = class_name
                break

        if matched_class is None:
            warnings.append(f"'{declared_item}' cannot be directly verified from the current packaging model.")
            continue

        if matched_class not in detections:
            missing_declared_items.append(declared_item)

    for flag_name, required in declared_flags.items():
        if required and flag_name not in detected_rules:
            missing_declared_items.append(flag_name.replace("_", " "))

    if declared_flags.get("shipping_label") and "shipping_label" not in detected_rules:
        blockers.append("A shipping label was declared but not detected in the video.")

    if declared_flags.get("fragile") and "fragile" not in detected_rules:
        blockers.append("Fragile handling was declared but not found in the video.")

    if declared_flags.get("keep_dry") and "keep_dry" not in detected_rules:
        blockers.append("Keep-dry handling was declared but not found in the video.")

    if declared_flags.get("protect_from_heat") and "protect_from_heat" not in detected_rules:
        blockers.append("Heat-protection handling was declared but not found in the video.")

    if declared_flags.get("handle_with_care") and "handle_with_care" not in detected_rules:
        blockers.append("Handle-with-care handling was declared but not found in the video.")

    if declared_flags.get("this_side_up") and "this_side_up" not in detected_rules:
        blockers.append("This-side-up marking was declared but not found in the video.")

    if blockers:
        verdict = "package is not suitable for shipping"
    elif missing_declared_items:
        verdict = "normal"
    else:
        verdict = "package can be shipped"

    if not declared_contents:
        warnings.append("No package contents were declared, so the model could only assess visible packaging cues.")

    if weight_kg <= 0:
        warnings.append("Package weight was not provided as a positive number; shipping feasibility may be incomplete.")

    return {
        "verdict": verdict,
        "warnings": warnings,
        "blockers": blockers,
        "missing_declared_items": sorted(set(missing_declared_items)),
        "detected_labels": detected_names,
        "detected_rules": sorted(detected_rules),
    }


app = FastAPI(title="Packaging Review Assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

mongo_client = build_mongo_client()
mongo_collection = mongo_client[MONGODB_DB]["analyses"] if mongo_client else None
model = load_yolo_model()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    try:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "model_ready": model is not None,
                "mongo_ready": mongo_collection is not None,
                "max_weight": MAX_SHIPPING_WEIGHT_KG,
                "max_seconds": MAX_VIDEO_SECONDS,
            },
        )
    except Exception:
        # Fallback: serve the raw HTML and substitute simple variables
        index_file = TEMPLATES_DIR / "index.html"
        if not index_file.exists():
            raise
        content = index_file.read_text(encoding="utf-8")
        content = content.replace("{{ max_seconds }}", str(MAX_VIDEO_SECONDS))
        content = content.replace("{{ max_weight }}", str(MAX_SHIPPING_WEIGHT_KG))
        content = content.replace("{{ 'yes' if model_ready else 'no' }}", "yes" if model is not None else "no")
        content = content.replace("{{ 'connected' if mongo_ready else 'not configured' }}", "connected" if mongo_collection is not None else "not configured")
        return HTMLResponse(content=content)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_ready": model is not None,
        "mongo_ready": mongo_collection is not None,
        "max_weight_kg": MAX_SHIPPING_WEIGHT_KG,
        "max_video_seconds": MAX_VIDEO_SECONDS,
    }


@app.post("/api/analyze")
async def analyze(
    video: UploadFile = File(...),
    package_weight_kg: float = Form(...),
    package_contents: str = Form(""),
    fragile: Optional[str] = Form(None),
    shipping_label: Optional[str] = Form(None),
    keep_dry: Optional[str] = Form(None),
    protect_from_heat: Optional[str] = Form(None),
    this_side_up: Optional[str] = Form(None),
    handle_with_care: Optional[str] = Form(None),
) -> JSONResponse:
    if not video.filename:
        raise HTTPException(status_code=400, detail="Please upload a video clip.")
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="YOLO model weights were not found. Set MODEL_WEIGHTS to the trained .pt file before analyzing packages.",
        )

    suffix = Path(video.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await video.read())

    try:
        video_analysis = analyze_video(temp_path, model)
        declared_contents = split_declared_items(package_contents)
        declared_flags = {
            "fragile": parse_bool(fragile),
            "shipping_label": parse_bool(shipping_label),
            "keep_dry": parse_bool(keep_dry),
            "protect_from_heat": parse_bool(protect_from_heat),
            "this_side_up": parse_bool(this_side_up),
            "handle_with_care": parse_bool(handle_with_care),
        }
        verdict_details = classify_submission(
            declared_contents=declared_contents,
            declared_flags=declared_flags,
            detections=video_analysis["detections"],
            weight_kg=package_weight_kg,
        )

        payload = {
            "created_at": datetime.now(timezone.utc),
            "package_weight_kg": package_weight_kg,
            "declared_contents": declared_contents,
            "declared_flags": declared_flags,
            "video_duration_seconds": video_analysis["duration_seconds"],
            "detections": video_analysis["detections"],
            "verdict": verdict_details["verdict"],
            "warnings": verdict_details["warnings"],
            "blockers": verdict_details["blockers"],
            "missing_declared_items": verdict_details["missing_declared_items"],
            "detected_labels": verdict_details["detected_labels"],
            "detected_rules": verdict_details["detected_rules"],
            "source_video_name": video.filename,
        }
        response_payload = {
            **payload,
            "created_at": payload["created_at"].isoformat(),
        }

        inserted_id = None
        if mongo_collection is not None:
            try:
                result = mongo_collection.insert_one(payload)
                inserted_id = str(result.inserted_id)
            except PyMongoError:
                inserted_id = None

        return JSONResponse({**response_payload, "stored": inserted_id is not None, "analysis_id": inserted_id, "model_ready": model is not None, "mongo_ready": mongo_collection is not None})
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
