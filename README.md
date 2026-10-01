# Peanut Mold Screening API

![CI](https://github.com/lindicaphxag-tech/peanut-aflatoxin-detection-api/actions/workflows/ci.yml/badge.svg)

A lightweight image-classification API for **visual peanut mold screening**. The service combines OpenCV preprocessing with a PyTorch ResNet18 classifier and exposes a small Flask REST API for inference.

> **Scope:** this project classifies visible mold-related image patterns into three categories. It is a software prototype for image-based screening and does **not** measure aflatoxin concentration or replace laboratory testing.

## What it does

- Accepts an uploaded image or base64-encoded image.
- Applies Otsu-based foreground masking before inference.
- Runs a 3-class ResNet18 classifier on CPU.
- Returns the predicted class, confidence, per-class probabilities, and aggregate mold probability.
- Uses request-scoped temporary files so concurrent requests do not overwrite one another.
- Lazy-loads PyTorch, OpenCV, torchvision, and the model only when inference is requested.
- Provides a lightweight health endpoint that works without loading the model.
- Includes API regression tests that run without downloading the model weights.

## Classes

| Index | Label |
| --- | --- |
| 0 | 正常 (normal) |
| 1 | 发霉不长毛 (moldy, no visible fuzz) |
| 2 | 发霉长毛 (moldy, visible fuzz) |

## Architecture

```text
HTTP request
    │
    ├─ validate upload / base64 payload
    ├─ request-scoped temporary file
    │
    ▼
lazy inference path
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

The model checkpoint is tracked with **Git LFS** (about 44.8 MB), so make sure LFS is installed before running inference.

```bash
git lfs install
git clone https://github.com/lindicaphxag-tech/peanut-aflatoxin-detection-api.git
cd peanut-aflatoxin-detection-api
git lfs pull

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

Example response before the first inference request:

```json
{
  "status": "ok",
  "model": "ResNet18-3class",
  "model_loaded": false
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

## Testing

The API layer is intentionally testable without loading the ML stack or downloading the checkpoint.

```bash
pip install Flask flask-cors pytest ruff
ruff check app.py tests
pytest -q
```

Current tests cover:

- health checks without model loading
- missing request payloads
- invalid base64 input
- empty image uploads
- successful request flow with mocked inference
- cleanup of request-scoped temporary files

## Stack

- Python
- Flask / Flask-CORS
- PyTorch / torchvision
- OpenCV
- Pillow
- NumPy
- Gunicorn
- pytest / Ruff
- GitHub Actions

## Repository layout

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── tests/
│   └── test_api.py
├── app.py
├── best_model_resnet18.pth   # Git LFS
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
