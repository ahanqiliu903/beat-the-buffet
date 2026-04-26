import csv
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from backend.app.llm import count_with_gpt4o
from backend.app.pricing import (
    DEFAULT_LOCATION,
    is_loaded,
    list_locations,
    load_prices,
    price_plate,
)
from backend.app import tables

load_dotenv()

CONFIG_PATH = "ml/configs/default.yaml"
STATIC_DIR = Path(__file__).parent / "static"

# Set ENABLE_DETECTOR=false in production to skip loading YOLO-World (~400 MB).
# GPT-4o handles all counting/labeling; the detector only draws decorative
# bounding boxes on the captured image.
ENABLE_DETECTOR = os.environ.get("ENABLE_DETECTOR", "true").lower() == "true"

ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,https://beat-the-buffet-six.vercel.app",
).split(",") if o.strip()]

state: dict = {}


def _load_display_names(labels_csv: str) -> dict[str, str]:
    with open(labels_csv, newline="") as f:
        return {row["class_name"]: row["display_name"] for row in csv.DictReader(f)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if ENABLE_DETECTOR:
        from ml.src.detector import build_detector
        state["detector"] = build_detector()
    else:
        print("[main] ENABLE_DETECTOR=false; skipping YOLO-World load")
    state["display"] = _load_display_names(cfg["data"]["labels_csv"])
    load_prices()
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/locations")
async def locations():
    return {"locations": list_locations(), "default": DEFAULT_LOCATION}


@app.post("/prewarm")
async def prewarm(location: str = Form(...)):
    already_cached = is_loaded(location)
    success = load_prices(location)
    return {"location": location, "loaded": success, "from_cache": already_cached}


class CreateTableRequest(BaseModel):
    table_name: str
    restaurant: str
    city: str
    ayce_price_per_person: float
    tax_included: bool
    tip_percent: float
    host_name: str


class JoinTableRequest(BaseModel):
    name: str


class TableCaptureRequest(BaseModel):
    participant_id: str
    total: int
    counts: list[dict]
    pricing: dict


@app.post("/tables")
async def create_table_endpoint(req: CreateTableRequest):
    table, participant_id = tables.create_table(
        table_name=req.table_name,
        restaurant=req.restaurant,
        city=req.city,
        ayce_price_per_person=req.ayce_price_per_person,
        tax_included=req.tax_included,
        tip_percent=req.tip_percent,
        host_name=req.host_name,
    )
    return {"table": table, "participant_id": participant_id}


@app.get("/tables/{code}")
async def get_table_endpoint(code: str):
    table = tables.get_table(code)
    if table is None:
        raise HTTPException(404, "Table not found")
    return table


@app.post("/tables/{code}/join")
async def join_table_endpoint(code: str, req: JoinTableRequest):
    table, participant_id = tables.join_table(code, req.name)
    if table is None:
        raise HTTPException(404, "Table not found")
    return {"table": table, "participant_id": participant_id}


@app.post("/tables/{code}/captures")
async def add_table_capture_endpoint(code: str, req: TableCaptureRequest):
    table = tables.add_capture(
        code, req.participant_id, req.total, req.counts, req.pricing,
    )
    if table is None:
        raise HTTPException(404, "Table or participant not found")
    return table


@app.post("/tables/{code}/finish")
async def finish_table_endpoint(code: str):
    table = tables.finish_table(code)
    if table is None:
        raise HTTPException(404, "Table not found")
    return table


@app.post("/identify")
async def identify(
    file: UploadFile = File(...),
    location: str = Form(DEFAULT_LOCATION),
):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = img.size

    box_coords: list[list[int]] = []
    if "detector" in state:
        from ml.src.detector import detect_pieces
        boxes = detect_pieces(state["detector"], img)
        box_coords = [[x1, y1, x2, y2] for x1, y1, x2, y2, _ in boxes]

    items, error = count_with_gpt4o(image_bytes, state["display"])
    if items is None:
        print(f"[llm] error: {error}")
        return {
            "image_size": [width, height],
            "boxes": box_coords,
            "counts": [],
            "total": 0,
            "error": error,
        }

    print(f"[llm] {items}")
    counts: list[dict] = []
    total = 0
    for item in items:
        name = item.get("name")
        qty = int(item.get("quantity", 0))
        if not name or qty <= 0:
            continue
        counts.append({
            "label": name,
            "display": state["display"].get(name, name),
            "count": qty,
        })
        total += qty

    counts.sort(key=lambda c: -c["count"])
    pricing = price_plate(counts, location)

    return {
        "image_size": [width, height],
        "boxes": box_coords,
        "counts": counts,
        "total": total,
        "pricing": pricing,
    }
