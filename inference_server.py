"""
inference_server.py — Python Flask model inference server
=========================================================
Loads the saved MobileNetV2 .keras model and exposes:

  POST /predict
    Body: multipart/form-data  with field  "image"  (jpg/png)
    Returns:
      {
        "isConstructionSite": true,
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

  Out-of-distribution detection
    The model was trained only on construction images, so it cannot explicitly
    say "this is not a construction site". We use Maximum Softmax Probability
    (MSP) thresholding: if max(softmax) < CONFIDENCE_THRESHOLD, the image is
    likely out-of-distribution (e.g. flowers, food, selfies) and
    isConstructionSite is set to False.

  GET /health
    Returns { "status": "ok", "modelLoaded": true }

Usage
-----
  cd model_training
  source venv/bin/activate   # (or: python3 -m venv venv && pip install flask tensorflow pillow flask-cors)
  python inference_server.py
"""

import io
import os
import time
import pathlib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Config ────────────────────────────────────────────────────────────────────
# Use path relative to THIS file so the server works regardless of launch directory
_HERE         = pathlib.Path(__file__).parent.resolve()
MODEL_PATH    = _HERE / "checkpoints" / "best_model.keras"
IMG_SIZE      = (224, 224)
MODEL_VERSION = "mobilenetv2-v1"
PORT          = int(os.environ.get("INFERENCE_PORT", 5001))

# Out-of-distribution threshold.
# If max(softmax) < this value the image is likely NOT a construction site.
#
# Why 0.80?  MobileNetV2 was trained ONLY on construction images, so it will
# produce softmax scores for any image — even waterfalls, selfies, or food.
# A well-trained model should score 85-95%+ on real construction photos.
# Raising this to 0.80 catches "confidently wrong" predictions on unrelated
# images (like waterfall→concrete_poured at 75%) while still passing real
# construction photos with a comfortable margin.
#
# The REAL fix is to retrain with non-construction (negative) examples so
# the model learns to say "I don't know" on out-of-domain images.
# Use prepare_negatives.py to download negatives, then re-run train.py.
CONFIDENCE_THRESHOLD = 0.80

CLASS_NAMES = [
    "formwork_erected",
    "reinforcement_placed",
    "concrete_poured",
    "structure_complete",
    "not_construction",   # added after retraining with diverse negative examples
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


# ── Fallback (model not loaded) ───────────────────────────────────────────────
def model_not_ready() -> dict:
    """
    Returns a zero-confidence result when the model hasn't been loaded.
    The Node.js backend treats 'dummy-random' modelVersion as invalid and
    ignores this result, so the report stays 'pending' rather than getting
    a fake score.
    """
    return {
        "detectedStage": None,
        "confidenceScore": 0.0,
        "allScores": {c: 0.0 for c in CLASS_NAMES},
        "modelVersion": "dummy-random",   # signals to backend: ignore this
        "inferenceMs": 0,
        "warning": "Model not loaded — run 'python train.py' first, then restart the server.",
    }


# ── Flask app ─────────────────────────────────────────────────────────────────
def create_app():
    app = Flask(__name__)
    CORS(app)  # allow Node.js backend on different port to call this

    # Pre-load on startup
    load_model()

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

        # ── If model not ready, return zero-confidence result ─────────────
        model = load_model()
        if model is None:
            return jsonify(model_not_ready()), 200

        # ── Inference ─────────────────────────────────────────────────────
        try:
            t0 = time.time()
            tensor = preprocess_image(file_bytes)
            preds = model.predict(tensor, verbose=0)[0]
            elapsed_ms = int((time.time() - t0) * 1000)
        except Exception as e:
            return jsonify({"error": f"Inference error: {str(e)}"}), 500

        idx = int(np.argmax(preds))
        max_score = float(preds[idx])
        scores = {c: round(float(s), 4) for c, s in zip(CLASS_NAMES, preds)}

        # ── Construction vs. not-construction detection ───────────────────
        # After retraining with a 'not_construction' class, we check two things:
        #   1. Explicit class prediction: if model picks 'not_construction', trust it.
        #   2. MSP threshold fallback: if confidence is below threshold, flag it.
        detected_class = CLASS_NAMES[idx]
        is_explicit_negative = (detected_class == "not_construction")
        is_construction = (not is_explicit_negative) and (max_score >= CONFIDENCE_THRESHOLD)

        if not is_construction:
            reason = (
                f"Model classified image as '{detected_class}' with {round(max_score * 100)}% confidence."
                if is_explicit_negative
                else f"Image does not appear to show a construction site "
                     f"(max confidence {round(max_score * 100)}% — below {round(CONFIDENCE_THRESHOLD * 100)}% threshold)."
            )
            return jsonify({
                "isConstructionSite": False,
                "detectedStage":      None,
                "confidenceScore":    round(max_score, 4),
                "allScores":          scores,
                "modelVersion":       MODEL_VERSION,
                "inferenceMs":        elapsed_ms,
                "observations":       reason,
            })

        return jsonify({
            "isConstructionSite": True,
            "detectedStage":      detected_class,
            "confidenceScore":    round(max_score, 4),
            "allScores":          scores,
            "modelVersion":       MODEL_VERSION,
            "inferenceMs":        elapsed_ms,
            "observations":       f"Model detected {detected_class} with {round(max_score * 100)}% confidence.",
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
            return jsonify(model_not_ready()), 200

        try:
            t0 = time.time()
            tensor = preprocess_image(file_bytes)
            preds = model.predict(tensor, verbose=0)[0]
            elapsed_ms = int((time.time() - t0) * 1000)
        except Exception as e:
            return jsonify({"error": f"Inference error: {str(e)}"}), 500

        idx = int(np.argmax(preds))
        max_score = float(preds[idx])
        scores = {c: round(float(s), 4) for c, s in zip(CLASS_NAMES, preds)}
        detected_class = CLASS_NAMES[idx]
        is_explicit_negative = (detected_class == "not_construction")
        is_construction = (not is_explicit_negative) and (max_score >= CONFIDENCE_THRESHOLD)

        if not is_construction:
            reason = (
                f"Model classified image as '{detected_class}' with {round(max_score * 100)}% confidence."
                if is_explicit_negative
                else f"Image does not appear to show a construction site "
                     f"(max confidence {round(max_score * 100)}% — below threshold)."
            )
            return jsonify({
                "isConstructionSite": False,
                "detectedStage":      None,
                "confidenceScore":    round(max_score, 4),
                "allScores":          scores,
                "modelVersion":       MODEL_VERSION,
                "inferenceMs":        elapsed_ms,
                "observations":       reason,
            })

        return jsonify({
            "isConstructionSite": True,
            "detectedStage":      detected_class,
            "confidenceScore":    round(max_score, 4),
            "allScores":          scores,
            "modelVersion":       MODEL_VERSION,
            "inferenceMs":        elapsed_ms,
            "observations":       f"Model detected {detected_class} with {round(max_score * 100)}% confidence.",
        })

    return app


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[INFO] Starting inference server on port {PORT}")
    app = create_app()
    app.run(host="0.0.0.0", port=PORT, debug=False)
