# Peanut Mold Screening API

A lightweight image-classification API for **visual peanut mold screening**. The service combines OpenCV preprocessing with a PyTorch ResNet18 classifier and exposes a small Flask REST API for inference.

> **Scope:** this project classifies visible mold-related image patterns into three categories. It is a software prototype for image-based screening and does **not** measure aflatoxin concentration or replace laboratory testing.

## What it does

- Accepts an uploaded image or base64-encoded image.
- Applies Otsu-based foreground masking before inference.
- Runs a 3-class ResNet18 classifier on CPU.
- Returns the predicted class, confidence, per-class probabilities, and an aggregate mold probability.
- Provides a health-check endpoint for deployment monitoring.

## Classes

| Index | Label |
| --- | --- |
| 0 | 正常 (normal) |
| 1 | 发霉不长毛 (moldy, no visible fuzz) |
| 2 | 发霉长毛 (moldy, visible fuzz) |

## Architecture

```text
Image
  │
  ├─ OpenCV decode
  ├─ grayscale + Otsu threshold
  ├─ foreground masking
  ├─ resize to 224×224
  └─ ImageNet normalization
          │
          ▼
      ResNet18
          │
          ▼
       Softmax
          │
          ▼
   JSON prediction
```

## Quick start

```bash
git clone https://github.com/lindicaphxag-tech/peanut-aflatoxin-detection-api.git
cd peanut-aflatoxin-detection-api
python -m venv .venv
```

Activate the environment, install dependencies, and start the service:

```bash
pip install -r requirements.txt
python app.py
```

The API listens on `http://127.0.0.1:5000` by default.

## API

### Health check

```http
GET /api/health
```

Example response:

```json
{
  "status": "ok",
  "model": "ResNet18-3class"
}
```

### Detect an image

```http
POST /api/detect
Content-Type: multipart/form-data
```

Form field: `image`

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/detect \
  -F "image=@sample.jpg"
```

The endpoint also accepts JSON containing a base64 payload:

```json
{
  "base64": "<encoded-image>"
}
```

A successful response contains:

```json
{
  "result": "正常",
  "confidence": 93.4,
  "normal_prob": 93.4,
  "moldy_light_prob": 4.1,
  "moldy_heavy_prob": 2.5,
  "moldy_prob": 6.6,
  "class_index": 0
}
```

## Stack

- Python
- Flask / Flask-CORS
- PyTorch / torchvision
- OpenCV
- Pillow
- NumPy
- Gunicorn

## Repository layout

```text
.
├── app.py
├── best_model_resnet18.pth
├── requirements.txt
└── Procfile
```

## Limitations

- The current service uses CPU inference.
- Predictions depend on the training distribution and image acquisition conditions.
- The repository currently does not include a reproducible training/evaluation pipeline or independently verified benchmark table.
- Visual mold classification is not equivalent to chemical aflatoxin quantification.

## Companion frontend

See [peanut-aflatoxin-detection-web](https://github.com/lindicaphxag-tech/peanut-aflatoxin-detection-web) for the React user interface.
