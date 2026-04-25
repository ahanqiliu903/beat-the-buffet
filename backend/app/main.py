import csv
import io
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend.app.llm import count_with_gpt4o
from backend.app.pricing import load_prices, price_plate
from ml.src.detector import build_detector, detect_pieces

load_dotenv()

CONFIG_PATH = "ml/configs/default.yaml"
STATIC_DIR = Path(__file__).parent / "static"

state: dict = {}


def _load_display_names(labels_csv: str) -> dict[str, str]:
    with open(labels_csv, newline="") as f:
        return {row["class_name"]: row["display_name"] for row in csv.DictReader(f)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    state["detector"] = build_detector()
    state["display"] = _load_display_names(cfg["data"]["labels_csv"])
    load_prices()
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = img.size

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
    pricing = price_plate(counts)

    return {
        "image_size": [width, height],
        "boxes": box_coords,
        "counts": counts,
        "total": total,
        "pricing": pricing,
    }
