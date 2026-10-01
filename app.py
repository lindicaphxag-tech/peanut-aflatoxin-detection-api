"""
Peanut mold screening backend API service.
"""

import base64
import os
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_resnet18.pth")
CLASS_NAMES = ["正常", "发霉不长毛", "发霉长毛"]

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

_model = None


def get_model():
    """Load and cache the classifier."""
    global _model
    if _model is None:
        print("正在加载模型...")
        _model = models.resnet18(weights=None)
        num_ftrs = _model.fc.in_features
        _model.fc = nn.Linear(num_ftrs, 3)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _model.eval()
        print("✅ 模型加载成功")
    return _model


def preprocess_image(image_path):
    """Apply foreground masking and model preprocessing."""
    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        img = Image.open(image_path).convert("RGB")
        return transform(img).unsqueeze(0)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB) / 255.0
    img_processed = (img_rgb * mask).astype(np.uint8)
    img_pil = Image.fromarray(img_processed)
    return transform(img_pil).unsqueeze(0)


def predict(image_path):
    """Run 3-class image classification."""
    model = get_model()
    tensor = preprocess_image(image_path)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()

    moldy_prob = probs[1].item() + probs[2].item()

    return {
        "result": CLASS_NAMES[pred_idx],
        "confidence": round(probs[pred_idx].item() * 100, 1),
        "normal_prob": round(probs[0].item() * 100, 1),
        "moldy_light_prob": round(probs[1].item() * 100, 1),
        "moldy_heavy_prob": round(probs[2].item() * 100, 1),
        "moldy_prob": round(moldy_prob * 100, 1),
        "class_index": pred_idx,
    }


def write_temp_image(image_bytes):
    """Write request bytes to a request-scoped temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_file.write(image_bytes)
        return temp_file.name


@app.route("/api/detect", methods=["POST"])
def detect():
    """Classify one uploaded or base64-encoded image."""
    temp_path = None
    try:
        if "image" in request.files:
            image_bytes = request.files["image"].read()
            if not image_bytes:
                return jsonify({"error": "上传的图片为空"}), 400
        else:
            payload = request.get_json(silent=True) or {}
            encoded = payload.get("base64")
            if not encoded:
                return jsonify({"error": "请上传图片"}), 400
            image_bytes = base64.b64decode(encoded, validate=True)

        temp_path = write_temp_image(image_bytes)
        return jsonify(predict(temp_path))

    except (ValueError, base64.binascii.Error):
        return jsonify({"error": "无效的Base64图片数据"}), 400
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "ResNet18-3class"})


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "name": "坚果霉变图像检测API",
            "version": "1.0.0",
            "endpoints": {
                "/api/detect": "POST - 上传图片检测",
                "/api/health": "GET - 健康检查",
            },
        }
    )


get_model()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
