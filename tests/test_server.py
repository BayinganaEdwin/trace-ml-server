"""
test_server.py — pytest tests for the Trace ML inference server (Flask)

Tests the Flask app using its built-in test client.
The model is not loaded in the test environment (no .keras file in temp dir),
so the server correctly falls back to dummy_predict() for /predict — this lets
us test all endpoint contracts without needing TensorFlow or the trained model.

Run from the model_training/ directory:
    pytest tests/ -v
"""

import io
import json
import os
import sys
import pytest
from PIL import Image

# ── Make sure inference_server is importable ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inference_server


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create Flask test client. Model is intentionally NOT loaded in tests —
    we redirect MODEL_PATH to a non-existent file so the server falls back to
    dummy_predict(), letting us test all endpoint contracts without TensorFlow.
    MODEL_PATH stays redirected for the full module scope so route handlers
    also skip the real checkpoint."""
    import pathlib
    inference_server._model = None
    # Keep MODEL_PATH redirected for the entire test module — route handlers
    # call load_model() on each request, so we must keep it pointing nowhere.
    inference_server.MODEL_PATH = pathlib.Path("/nonexistent/model.keras")

    app = inference_server.create_app()
    app.config["TESTING"] = True

    with app.test_client() as c:
        yield c


def make_image_bytes(width=224, height=224, color=(120, 180, 90)) -> bytes:
    """Create a minimal valid JPEG in memory for upload tests."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ── /health ──────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200

    def test_health_returns_ok_status(self, client):
        res = client.get("/health")
        data = res.get_json()
        assert data["status"] == "ok"

    def test_health_includes_class_list(self, client):
        res = client.get("/health")
        data = res.get_json()
        assert "classes" in data
        assert "formwork_erected" in data["classes"]
        assert "not_construction" in data["classes"]
        assert len(data["classes"]) == 5

    def test_health_reports_model_loaded_field(self, client):
        res = client.get("/health")
        data = res.get_json()
        assert "modelLoaded" in data
        # In test env the model is not loaded
        assert data["modelLoaded"] is False


# ── /predict ─────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_returns_200_with_valid_jpeg(self, client):
        img_bytes = make_image_bytes()
        data = {"image": (io.BytesIO(img_bytes), "photo.jpg")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
        assert res.status_code == 200

    def test_predict_response_has_required_fields(self, client):
        img_bytes = make_image_bytes()
        data = {"image": (io.BytesIO(img_bytes), "photo.jpg")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
        body = res.get_json()
        assert "detectedStage" in body
        assert "confidenceScore" in body
        assert "allScores" in body
        assert "modelVersion" in body

    def test_predict_detected_stage_is_valid_class(self, client):
        VALID_CLASSES = {
            "formwork_erected", "reinforcement_placed",
            "concrete_poured", "structure_complete", "not_construction"
        }
        img_bytes = make_image_bytes()
        data = {"image": (io.BytesIO(img_bytes), "photo.jpg")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
        body = res.get_json()
        assert body["detectedStage"] in VALID_CLASSES

    def test_predict_confidence_score_is_between_0_and_1(self, client):
        img_bytes = make_image_bytes()
        data = {"image": (io.BytesIO(img_bytes), "photo.jpg")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
        body = res.get_json()
        assert 0.0 <= body["confidenceScore"] <= 1.0

    def test_predict_all_scores_sum_to_approximately_1(self, client):
        img_bytes = make_image_bytes()
        data = {"image": (io.BytesIO(img_bytes), "photo.jpg")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
        body = res.get_json()
        total = sum(body["allScores"].values())
        assert abs(total - 1.0) < 0.05  # allow small floating point variance

    def test_predict_returns_400_when_image_field_missing(self, client):
        res = client.post("/predict", data={}, content_type="multipart/form-data")
        assert res.status_code == 400
        body = res.get_json()
        assert "error" in body

    def test_predict_returns_415_for_unsupported_file_type(self, client):
        data = {"image": (io.BytesIO(b"fake pdf content"), "document.pdf")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
        assert res.status_code == 415

    def test_predict_accepts_png_image(self, client):
        img = Image.new("RGB", (224, 224), color=(50, 100, 150))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = {"image": (io.BytesIO(buf.getvalue()), "photo.png")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
        assert res.status_code == 200


# ── /compare ─────────────────────────────────────────────────────────────────

class TestCompare:
    def test_compare_returns_200_with_two_valid_images(self, client):
        img_a = make_image_bytes(color=(100, 100, 100))
        img_b = make_image_bytes(color=(200, 200, 200))
        data = {
            "image_a": (io.BytesIO(img_a), "prev.jpg"),
            "image_b": (io.BytesIO(img_b), "curr.jpg"),
        }
        res = client.post("/compare", data=data, content_type="multipart/form-data")
        assert res.status_code == 200

    def test_compare_response_has_required_fields(self, client):
        img_a = make_image_bytes(color=(100, 100, 100))
        img_b = make_image_bytes(color=(200, 200, 200))
        data = {
            "image_a": (io.BytesIO(img_a), "prev.jpg"),
            "image_b": (io.BytesIO(img_b), "curr.jpg"),
        }
        res = client.post("/compare", data=data, content_type="multipart/form-data")
        body = res.get_json()
        assert "similarityScore" in body
        assert "progressDelta" in body
        assert "progressLabel" in body
        assert "stagnant" in body

    def test_compare_identical_images_marked_stagnant(self, client):
        img_bytes = make_image_bytes(color=(128, 128, 128))
        data = {
            "image_a": (io.BytesIO(img_bytes), "prev.jpg"),
            "image_b": (io.BytesIO(img_bytes), "curr.jpg"),  # exact same image
        }
        res = client.post("/compare", data=data, content_type="multipart/form-data")
        body = res.get_json()
        assert body["stagnant"] is True
        assert body["progressDelta"] < 0.05

    def test_compare_returns_400_when_one_image_missing(self, client):
        img_bytes = make_image_bytes()
        data = {"image_a": (io.BytesIO(img_bytes), "prev.jpg")}  # missing image_b
        res = client.post("/compare", data=data, content_type="multipart/form-data")
        assert res.status_code == 400

    def test_compare_progress_delta_is_between_0_and_1(self, client):
        img_a = make_image_bytes(color=(50, 50, 50))
        img_b = make_image_bytes(color=(200, 200, 200))
        data = {
            "image_a": (io.BytesIO(img_a), "prev.jpg"),
            "image_b": (io.BytesIO(img_b), "curr.jpg"),
        }
        res = client.post("/compare", data=data, content_type="multipart/form-data")
        body = res.get_json()
        assert 0.0 <= body["progressDelta"] <= 1.0
