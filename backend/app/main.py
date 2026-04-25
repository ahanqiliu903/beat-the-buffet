import io
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from ml.src.dataset import build_transforms, load_labels
from ml.src.model import build_model

CONFIG_PATH = "ml/configs/default.yaml"
CHECKPOINT_PATH = "ml/checkpoints/best.pt"
STATIC_DIR = Path(__file__).parent / "static"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    labels = load_labels(cfg["data"]["labels_csv"])
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    state["model"] = model
    state["device"] = device
    state["transform"] = build_transforms(cfg["data"]["image_size"], train=False, aug=cfg["augment"])
    state["id_to_name"] = {v: k for k, v in labels.items()}
    state["display"] = _load_display_names(cfg["data"]["labels_csv"])
    yield
    state.clear()


def _load_display_names(labels_csv: str) -> dict[str, str]:
    import csv
    out = {}
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            out[row["class_name"]] = row["display_name"]
    return out


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/identify")
async def identify(file: UploadFile = File(...), top_k: int = 4):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = state["transform"](img).unsqueeze(0).to(state["device"])
    with torch.no_grad():
        probs = torch.softmax(state["model"](tensor), dim=1)[0]
    k = min(top_k, probs.size(0))
    top = torch.topk(probs, k=k)
    return {
        "predictions": [
            {
                "label": state["id_to_name"][int(i)],
                "display": state["display"][state["id_to_name"][int(i)]],
                "confidence": float(p),
            }
            for p, i in zip(top.values, top.indices)
        ]
    }
