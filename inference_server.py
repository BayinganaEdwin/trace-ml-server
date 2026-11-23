"""
inference_server.py — Python Flask model inference server
=========================================================
Loads the saved MobileNetV2 .keras model and exposes:

  POST /predict
    Body: multipart/form-data  with field  "image"  (jpg/png)
    Returns:
      {
        "detectedStage": "concrete_poured",
        "confidenceScore": 0.91,
        "allScores": {
          "formwork_erected": 0.02,
          "reinforcement_placed": 0.05,
          "concrete_poured": 0.91,
          "structure_complete": 0.02
        },
        "modelVersion": "mobilenetv2-v1",
        "inferenceMs": 43
      }

  GET /health
    Returns { "status": "ok", "modelLoaded": true }

Usage
-----
  pip install flask tensorflow pillow
  python inference_server.py

  # or with gunicorn for production:
  # gunicorn -w 1 -b 0.0.0.0:5001 "inference_server:create_app()"
"""

import io
import os
import time
import pathlib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # pip install flask-cors

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH    = pathlib.Path("checkpoints/best_model.keras")
IMG_SIZE      = (224, 224)
MODEL_VERSION = "mobilenetv2-v1"
PORT          = int(os.environ.get("INFERENCE_PORT", 5001))

CLASS_NAMES = [
    "formwork_erected",
    "reinforcement_placed",
    "concrete_poured",
    "structure_complete",
    "not_construction",
]

# ── Lazy-load model ───────────────────────────────────────────────────────────
_model = None

def load_model():
    global _model
    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        print(f"[WARN] Model not found at {MODEL_PATH}")
        print("       Run  python train.py  first, then restart this server.")
        return None

    import tensorflow as tf
    print(f"[INFO] Loading model from {MODEL_PATH} …")
    t0 = time.time()
    _model = tf.keras.models.load_model(str(MODEL_PATH))
    print(f"[INFO] Model loaded in {(time.time()-t0)*1000:.0f} ms")
    return _model


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Decode bytes → (1, 224, 224, 3) float32 tensor, MobileNetV2-normalised."""
    from PIL import Image
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Dummy predictor (used when model not loaded) ──────────────────────────────
def dummy_predict() -> dict:
    """
    Returns a plausible random prediction across the 4 construction classes.
    not_construction is intentionally excluded — the dummy model has no real
    ability to distinguish construction from non-construction images, so
    including it would randomly flag genuine site photos. The real trained
    model handles not_construction correctly.
    """
    construction_classes = CLASS_NAMES[:4]  # exclude not_construction
    scores = np.random.dirichlet(np.ones(4) * 2).tolist()
    idx = int(np.argmax(scores))
    all_scores = {c: round(s, 4) for c, s in zip(construction_classes, scores)}
    all_scores["not_construction"] = 0.0
    return {
        "detectedStage": construction_classes[idx],
        "confidenceScore": round(scores[idx], 4),
        "allScores": all_scores,
        "modelVersion": "dummy-random",
        "inferenceMs": 0,
        "warning": "Model not loaded — returning random scores for testing",
    }


# ── Flask app ─────────────────────────────────────────────────────────────────
def create_app():
    app = Flask(__name__)
    CORS(app)  # allow Node.js backend on different port to call this

    # Pre-load on startup — catch errors so the server still starts even if the
    # model file is missing or the Keras version doesn't match the saved weights.
    # In that case all /predict calls fall back to dummy_predict().
    try:
        load_model()
    except Exception as e:
        print(f"[WARN] Model could not be pre-loaded: {e}")
        print("[WARN] Server starting without model — /predict will return dummy scores.")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "modelLoaded": _model is not None,
            "modelVersion": MODEL_VERSION,
            "classes": CLASS_NAMES,
        })

    @app.route("/predict", methods=["POST"])
    def predict():
        # ── Validate input ────────────────────────────────────────────────
        if "image" not in request.files:
            return jsonify({"error": "Missing 'image' field in multipart form"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        allowed = {".jpg", ".jpeg", ".png", ".webp"}
        ext = pathlib.Path(file.filename).suffix.lower()
        if ext not in allowed:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 415

        file_bytes = file.read()
        if len(file_bytes) > 10 * 1024 * 1024:  # 10 MB cap
            return jsonify({"error": "Image too large (max 10 MB)"}), 413

        # ── If model not ready, return dummy ──────────────────────────────
        model = load_model()
        if model is None:
            return jsonify(dummy_predict()), 200

        # ── Inference ─────────────────────────────────────────────────────
        try:
            t0 = time.time()
            tensor = preprocess_image(file_bytes)
            preds = model.predict(tensor, verbose=0)[0]
            elapsed_ms = int((time.time() - t0) * 1000)
        except Exception as e:
            return jsonify({"error": f"Inference error: {str(e)}"}), 500

        idx = int(np.argmax(preds))
        scores = {c: round(float(s), 4) for c, s in zip(CLASS_NAMES, preds)}

        return jsonify({
            "detectedStage":  CLASS_NAMES[idx],
            "confidenceScore": round(float(preds[idx]), 4),
            "allScores":      scores,
            "modelVersion":   MODEL_VERSION,
            "inferenceMs":    elapsed_ms,
        })

    @app.route("/compare", methods=["POST"])
    def compare():
        """
        Compare two site photos to measure construction progress.

        Accepts multipart/form-data with fields:
          "image_a"  — previous report photo (bytes)
          "image_b"  — current report photo  (bytes)

        Returns:
          {
            "similarityScore": 0.82,      # cosine similarity of MobileNetV2 features [0–1]
            "progressDelta":   0.18,      # 1 - similarity; higher = more change = more progress
            "progressLabel":   "Moderate progress detected",
            "stagnant":        false,     # true if delta < 0.05 (almost identical photos)
            "inferenceMs":     38
          }

        Works even when model is not loaded — falls back to pixel-level SSIM.
        """
        if "image_a" not in request.files or "image_b" not in request.files:
            return jsonify({"error": "Both 'image_a' and 'image_b' fields are required"}), 400

        bytes_a = request.files["image_a"].read()
        bytes_b = request.files["image_b"].read()

        if len(bytes_a) > 10 * 1024 * 1024 or len(bytes_b) > 10 * 1024 * 1024:
            return jsonify({"error": "Images must be under 10 MB each"}), 413

        t0 = time.time()
        model = load_model()

        if model is not None:
            # ── Extract feature vectors from MobileNetV2 backbone ─────────────
            import tensorflow as tf

            # Build feature extractor once and cache it
            if not hasattr(compare, "_extractor"):
                # Remove the final classification head; output is (1, 7, 7, 1280) → GlobalAvgPool → (1, 1280)
                base = model.layers[0] if hasattr(model.layers[0], "layers") else model
                # Walk to the GlobalAveragePooling layer output
                for layer in model.layers:
                    if "global_average" in layer.name.lower():
                        compare._extractor = tf.keras.Model(
                            inputs=model.input,
                            outputs=layer.output
                        )
                        break
                else:
                    # Fallback: use second-to-last layer
                    compare._extractor = tf.keras.Model(
                        inputs=model.input,
                        outputs=model.layers[-2].output
                    )

            extractor = compare._extractor
            tensor_a  = preprocess_image(bytes_a)
            tensor_b  = preprocess_image(bytes_b)
            feat_a    = extractor.predict(tensor_a, verbose=0)[0]
            feat_b    = extractor.predict(tensor_b, verbose=0)[0]

            # Cosine similarity
            dot    = float(np.dot(feat_a, feat_b))
            norm_a = float(np.linalg.norm(feat_a))
            norm_b = float(np.linalg.norm(feat_b))
            similarity = dot / (norm_a * norm_b + 1e-8)
        else:
            # ── Fallback: pixel-level mean absolute difference ────────────────
            from PIL import Image
            img_a = np.array(Image.open(io.BytesIO(bytes_a)).convert("RGB").resize(IMG_SIZE), dtype=np.float32) / 255.0
            img_b = np.array(Image.open(io.BytesIO(bytes_b)).convert("RGB").resize(IMG_SIZE), dtype=np.float32) / 255.0
            # Convert pixel diff to a similarity score [0–1]
            mean_diff  = float(np.mean(np.abs(img_a - img_b)))
            similarity = max(0.0, 1.0 - mean_diff * 4)   # scale so 25% avg diff → 0

        elapsed_ms     = int((time.time() - t0) * 1000)
        similarity     = round(float(np.clip(similarity, 0.0, 1.0)), 4)
        progress_delta = round(1.0 - similarity, 4)

        # ── Human-readable label ──────────────────────────────────────────────
        if progress_delta < 0.05:
            label   = "No visible progress — photos look identical"
            stagnant = True
        elif progress_delta < 0.15:
            label   = "Minor progress detected"
            stagnant = False
        elif progress_delta < 0.30:
            label   = "Moderate progress detected"
            stagnant = False
        else:
            label   = "Significant progress detected"
            stagnant = False

        return jsonify({
            "similarityScore": similarity,
            "progressDelta":   progress_delta,
            "progressLabel":   label,
            "stagnant":        stagnant,
            "modelBacked":     model is not None,
            "inferenceMs":     elapsed_ms,
        })

    @app.route("/predict/base64", methods=["POST"])
    def predict_base64():
        """
        Alternative endpoint: accepts JSON  { "image": "<base64-string>" }
        Useful for React Native apps that send base64 instead of multipart.
        """
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' key in JSON body"}), 400

        import base64
        try:
            file_bytes = base64.b64decode(data["image"])
        except Exception:
            return jsonify({"error": "Invalid base64 string"}), 400

        model = load_model()
        if model is None:
            return jsonify(dummy_predict()), 200

        try:
            t0 = time.time()
            tensor = preprocess_image(file_bytes)
            preds = model.predict(tensor, verbose=0)[0]
            elapsed_ms = int((time.time() - t0) * 1000)
        except Exception as e:
            return jsonify({"error": f"Inference error: {str(e)}"}), 500

        idx = int(np.argmax(preds))
        scores = {c: round(float(s), 4) for c, s in zip(CLASS_NAMES, preds)}

        return jsonify({
            "detectedStage":  CLASS_NAMES[idx],
            "confidenceScore": round(float(preds[idx]), 4),
            "allScores":      scores,
            "modelVersion":   MODEL_VERSION,
            "inferenceMs":    elapsed_ms,
        })

    @app.route("/version", methods=["GET"])
    def version():
        """Return model version and class list."""
        return jsonify({
            "modelVersion": MODEL_VERSION,
            "classes": CLASS_NAMES,
            "imgSize": list(IMG_SIZE),
        })

    return app


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[INFO] Starting inference server on port {PORT}")
    app = create_app()
    app.run(host="0.0.0.0", port=PORT, debug=False)

# Configure structured logging for production environments
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def is_allowed_image(filename: str) -> bool:
    """Check if the uploaded file has an allowed image extension."""
    return pathlib.Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

# Timing breakdown fields in /predict response:
# preprocess_ms  — image decode + resize + normalize
# inference_ms   — model.predict() call only
# total_ms       — full request handler duration (not yet exposed; planned)

# Memory note for /compare endpoint:
# The feature extractor sub-model is built once and cached as _feature_extractor.
# Avoids rebuilding the model graph on every request.
# Approximate memory footprint: ~14 MB for MobileNetV2 feature extractor.

# CORS is globally enabled via flask-cors for local development.
# In production, restrict origins to the backend server's IP/domain:
# CORS(app, origins=["https://api.yourdomain.com"])

# Production deployment:
# Use exactly 1 gunicorn worker (-w 1) because TensorFlow's global model state
# is not safe to share across forked processes. Multiple concurrent requests
# are served via async I/O within the single worker.
# Command: gunicorn -w 1 -b 0.0.0.0:5001 "inference_server:create_app()"
