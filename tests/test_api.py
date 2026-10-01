import base64
import io
import os

import pytest

import app as api


@pytest.fixture()
def client():
    api.app.config.update(TESTING=True)
    return api.app.test_client()


def test_health_does_not_load_model(client):
    api._model = None

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "ok",
        "model": "ResNet18-3class",
        "model_loaded": False,
    }
    assert api._model is None


def test_detect_rejects_missing_payload(client):
    response = client.post("/api/detect", json={})

    assert response.status_code == 400
    assert response.get_json()["error"] == "请上传图片"


def test_detect_rejects_invalid_base64(client):
    response = client.post("/api/detect", json={"base64": "%%%not-base64%%%"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "无效的Base64图片数据"


def test_detect_rejects_empty_upload(client):
    response = client.post(
        "/api/detect",
        data={"image": (io.BytesIO(b""), "empty.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "上传的图片为空"


def test_detect_cleans_up_request_scoped_temp_file(client, monkeypatch):
    seen_paths = []

    def fake_predict(image_path):
        seen_paths.append(image_path)
        assert os.path.exists(image_path)
        with open(image_path, "rb") as image_file:
            assert image_file.read() == b"fake-image"
        return {
            "result": "正常",
            "confidence": 99.0,
            "normal_prob": 99.0,
            "moldy_light_prob": 0.5,
            "moldy_heavy_prob": 0.5,
            "moldy_prob": 1.0,
            "class_index": 0,
        }

    monkeypatch.setattr(api, "predict", fake_predict)
    encoded = base64.b64encode(b"fake-image").decode("ascii")

    response = client.post("/api/detect", json={"base64": encoded})

    assert response.status_code == 200
    assert response.get_json()["class_index"] == 0
    assert len(seen_paths) == 1
    assert not os.path.exists(seen_paths[0])
