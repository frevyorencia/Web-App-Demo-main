"""Streamlit tutorial: NVIDIA Grounding DINO object detection via API."""
# --- Imports ---------------------------------------------------------------
import io
import json
import os
import time
import uuid
import zipfile
from typing import Dict, List, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# --- Streamlit Page Setup --------------------------------------------------
st.set_page_config(page_title="Andy's Detector", page_icon="🔍", layout="centered")
st.title("Andy's Detector")

# Simple inputs so learners can follow along
prompt = st.text_input("Prompt", value="find all objects")
uploaded_file = st.file_uploader(
    "Image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=False,
)

if uploaded_file:
    st.image(uploaded_file, use_container_width=True)

if "result_image" not in st.session_state:
    st.session_state["result_image"] = None
if "detections" not in st.session_state:
    st.session_state["detections"] = []


# --- Section 1: API + Deep Learning Utilities -----------------------------
# These constants mirror NVIDIA's official example.
NVAI_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nv-grounding-dino"
NVAI_POLLING_URL = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
ASSETS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
UPLOAD_ASSET_TIMEOUT = 300
MAX_RETRIES = 5
DELAY_BTW_RETRIES = 1
def _auth_value() -> str:
    """Read the API token from Streamlit secrets or environment."""
    return (
        st.secrets.get("NVIDIA_API_KEY")
        or st.secrets.get("API_KEY")
        or os.getenv("NVIDIA_PERSONAL_API_KEY")
        or os.getenv("NGC_PERSONAL_API_KEY")
        or ""
    )
def _upload_asset(data: bytes, description: str, content_type: str) -> str:
    """Upload the raw image to NVIDIA's asset storage and return its ID."""
    headers = {"Content-Type": "application/json", "accept": "application/json"}
    auth = _auth_value()
    if auth:
        headers["Authorization"] = f"Bearer {auth}"

    payload = {"contentType": content_type, "description": description}
    response = requests.post(ASSETS_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    meta = response.json()
    upload_url = meta["uploadUrl"]
    asset_id = str(uuid.UUID(meta["assetId"]))

    s3_headers = {
        "x-amz-meta-nvcf-asset-description": description,
        "content-type": content_type,
    }
    response = requests.put(upload_url, data=data, headers=s3_headers, timeout=UPLOAD_ASSET_TIMEOUT)
    response.raise_for_status()
    return asset_id
def _request_zip(asset_id: str, prompt_text: str, content_type: str) -> bytes:
    """Call nv-grounding-dino and return the zipped response bytes."""
    auth = _auth_value()
    headers = {
        "Content-Type": "application/json",
        "NVCF-INPUT-ASSET-REFERENCES": asset_id,
        "NVCF-FUNCTION-ASSET-IDS": asset_id,
    }
    if auth:
        headers["Authorization"] = f"Bearer {auth}"

    payload = {
        "model": "Grounding-Dino",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "media_url",
                        "media_url": {"url": f"data:{content_type};asset_id,{asset_id}"},
                    },
                ],
            }
        ],
        "threshold": 0.3,
    }

    response = requests.post(NVAI_URL, headers=headers, json=payload, timeout=120)
    if response.status_code == 200:
        return response.content
    if response.status_code == 202:
        reqid = response.headers.get("NVCF-REQID", "")
        poll_headers = {"accept": "application/json"}
        if auth:
            poll_headers["Authorization"] = f"Bearer {auth}"
        poll_url = f"{NVAI_POLLING_URL}{reqid}"
        retries = MAX_RETRIES
        while retries > 0:
            time.sleep(DELAY_BTW_RETRIES)
            poll = requests.get(poll_url, headers=poll_headers, timeout=120)
            if poll.status_code == 200:
                return poll.content
            if poll.status_code != 202:
                break
            retries -= 1
        return b""
    response.raise_for_status()
    return response.content
def _extract_detections(data, width: int, height: int) -> List[Dict[str, float]]:
    """Normalize detection JSON into label, score, and pixel bbox."""
    items: List[Dict[str, float]] = []
    if isinstance(data, dict):
        for key in ("predictions", "detections", "objects", "results", "data"):
            if isinstance(data.get(key), list):
                items = data[key]
                break
        else:
            items = [data]
    elif isinstance(data, list):
        items = data

    detections: List[Dict[str, float]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        bbox_info = item.get("bbox") or item.get("box") or item.get("bounding_box") or {}
        bbox = _to_pixels(bbox_info, width, height)
        detections.append(
            {
                "label": str(item.get("label") or item.get("class") or item.get("text") or "object"),
                "confidence": float(item.get("confidence") or item.get("score") or 0),
                "bbox": bbox,
            }
        )
    return detections
def _parse_zip(zip_bytes: bytes, width: int, height: int) -> Tuple[List[Dict[str, float]], bytes]:
    """Read detections and annotated image from the returned zip file."""
    if not zip_bytes:
        return [], b""

    detections: List[Dict[str, float]] = []
    annotated: bytes = b""

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            lower = name.lower()
            if not annotated and lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
                annotated = zf.read(name)
            if lower.endswith(".json"):
                try:
                    data = json.loads(zf.read(name).decode("utf-8"))
                except Exception:  # tutorial keeps handling simple
                    continue
                detections.extend(_extract_detections(data, width, height))

    return detections, annotated
def _to_pixels(box: Dict[str, float], width: int, height: int) -> List[float]:
    """Convert normalized or absolute bbox values into pixels."""
    x = float(box.get("x") or box.get("xmin") or 0)
    y = float(box.get("y") or box.get("ymin") or 0)
    w = float(box.get("width") or (box.get("xmax", 0) - x))
    h = float(box.get("height") or (box.get("ymax", 0) - y))
    if 0 <= x <= 1 and 0 <= w <= 1:
        x *= width
        w *= width
    if 0 <= y <= 1 and 0 <= h <= 1:
        y *= height
        h *= height
    return [x, y, x + w, y + h]
def annotate(image_bytes: bytes, detections: List[Dict[str, float]]) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = ["#f97316", "#2563eb", "#16a34a", "#db2777", "#7c3aed"]

    for idx, item in enumerate(detections):
        bbox = item.get("bbox")
        if not bbox:
            continue
        x_min, y_min, x_max, y_max = bbox
        color = colors[idx % len(colors)]
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

        text = item.get("label", "object")
        score = item.get("confidence")
        if score is not None:
            text += f" {score:.2f}"
        text_box = draw.textbbox((0, 0), text, font=font)
        padding = 3
        background = [
            x_min,
            max(0, y_min - (text_box[3] - text_box[1]) - padding * 2),
            x_min + (text_box[2] - text_box[0]) + padding * 2,
            max((text_box[3] - text_box[1]) + padding * 2, y_min),
        ]
        draw.rectangle(background, fill=color)
        draw.text((background[0] + padding, background[1] + padding), text, fill="white", font=font)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
def detect(image_bytes: bytes, prompt_text: str, content_type: str) -> Tuple[List[Dict[str, float]], bytes]:
    """High-level helper: upload, request detections, unpack results."""
    content_type = content_type or "image/png"
    asset_id = _upload_asset(image_bytes, "Input Asset", content_type)
    zip_bytes = _request_zip(asset_id, prompt_text, content_type)
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size
    detections, annotated_bytes = _parse_zip(zip_bytes, width, height)
    if not annotated_bytes:  # fallback when API zip has no image preview inside
        annotated_bytes = annotate(image_bytes, detections)
    return detections, annotated_bytes

# --- Section 2: Web UI Wiring ---------------------------------------------
# This layer calls the helpers above and shows results in Streamlit.
if st.button("Run Detection") and uploaded_file and prompt.strip():
    image_bytes = uploaded_file.getvalue()
    detections, annotated_bytes = detect(image_bytes, prompt.strip(), uploaded_file.type)
    st.session_state["detections"] = detections
    st.session_state["result_image"] = annotated_bytes

if st.session_state.get("result_image"):
    st.image(Image.open(io.BytesIO(st.session_state["result_image"])), use_container_width=True)
    for item in st.session_state.get("detections", []):
        label = item.get("label", "object")
        score = item.get("confidence")
        bbox = item.get("bbox")
        line = f"- {label}"
        if score is not None:
            line += f" ({score:.2f})"
        if bbox:
            line += f" bbox={[round(v, 1) for v in bbox]}"
        st.write(line)