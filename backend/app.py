"""
Flask Backend API for AI-Based Fake Social Media Account Detection System
--------------------------------------------------------------------------
Endpoints:
  GET  /                    -> Serves the Frontend Web App
  GET  /api/health          -> API Health Check & System Status
  GET  /api/metrics         -> Returns Model Performance & Dataset Metrics
  POST /api/predict         -> Predicts Account Authenticity & Risk Score (manual form)
  POST /api/analyze-account -> URL-Based Platform Analysis (Instagram / Facebook / X)
"""

import os
import sys
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.model import FakeAccountPredictor
from backend.utils.url_validator import validate_and_extract
from backend.utils.feature_processor import process_features
from backend.extractors.instagram_extractor import extract_instagram
from backend.extractors.facebook_extractor import extract_facebook
from backend.extractors.x_extractor import extract_x

# Initialize Flask app
app = Flask(__name__, static_folder=FRONTEND_DIR)
# Enable CORS for all routes to support standalone frontend development / Live Server
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize Predictor singleton
predictor = None
try:
    predictor = FakeAccountPredictor()
    print("[+] Model loaded successfully into Flask backend.")
except Exception as e:
    print(f"[!] Warning: Could not initialize model at startup: {e}")
    predictor = None

# ----------------------------------------------------------------------
# Frontend Serving Routes
# ----------------------------------------------------------------------
@app.route("/")
def serve_index():
    """Serves the main application homepage."""
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def serve_static_files(filename):
    """Serves static assets (CSS, JS, images)."""
    return send_from_directory(FRONTEND_DIR, filename)

# ----------------------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify backend status and model readiness.
    """
    is_ready = predictor is not None and predictor.model is not None
    return jsonify({
        "status": "healthy" if is_ready else "degraded",
        "service": "AI-Based Fake Social Media Account Detection System",
        "version": "1.0.0",
        "model_loaded": is_ready,
        "dataset_type": predictor.metrics.get("dataset_type", "synthetic_demonstration_dataset") if predictor else "unknown",
        "is_demonstration": predictor.metrics.get("is_demonstration", True) if predictor else True,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200

@app.route("/api/metrics", methods=["GET"])
def get_model_metrics():
    """
    Returns true evaluation metrics and feature importances calculated on the test set.
    """
    if not predictor or not predictor.metrics:
        return jsonify({
            "success": False,
            "error": "Model metrics are not available. Please ensure model training was executed."
        }), 503

    return jsonify({
        "success": True,
        "metrics": predictor.metrics
    }), 200

@app.route("/api/predict", methods=["POST"])
def predict_account():
    """
    Accepts account features in JSON, validates, and runs Random Forest inference.
    """
    if not predictor or not predictor.model:
        return jsonify({
            "success": False,
            "error": "Machine learning model is not loaded. Train the model using 'python backend/train_model.py'."
        }), 503

    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request payload must be in application/json format."
        }), 400

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({
            "success": False,
            "error": "Malformed JSON payload provided."
        }), 400

    # Run inference pipeline
    result = predictor.predict(payload)

    if not result.get("success"):
        return jsonify({
            "success": False,
            "error": "Validation failed for submitted account features.",
            "details": result.get("errors", [])
        }), 400

    return jsonify(result), 200

# ----------------------------------------------------------------------
# Global Error Handlers
# ----------------------------------------------------------------------
@app.errorhandler(404)
def handle_not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "API endpoint not found"}), 404
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.errorhandler(500)
def handle_internal_error(e):
    return jsonify({
        "success": False,
        "error": "Internal server error occurred.",
        "message": str(e)
    }), 500

@app.route("/api/analyze-account", methods=["POST"])
def analyze_account_by_url():
    """
    URL-based social media account analysis.

    Accepts:
      { "platform": "instagram"|"facebook"|"x", "profile_url": "https://..." }

    Returns:
      Extracted account info + ML prediction (or insufficient_data flag).
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be application/json."}), 400

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"success": False, "error": "Malformed JSON payload."}), 400

    platform_hint = str(payload.get("platform", "")).strip().lower()
    profile_url   = str(payload.get("profile_url", "")).strip()

    if not profile_url:
        return jsonify({"success": False, "error": "profile_url is required."}), 400

    if platform_hint not in ("instagram", "facebook", "x", "twitter", ""):
        return jsonify({
            "success": False,
            "error": "platform must be 'instagram', 'facebook', or 'x'."
        }), 400

    # ------------------------------------------------------------------ #
    # Step 1: Validate URL & extract username
    # ------------------------------------------------------------------ #
    validation = validate_and_extract(profile_url)
    if not validation["success"]:
        return jsonify({
            "success": False,
            "error":   validation["error"],
            "step":    "url_validation"
        }), 400

    platform = validation["platform"]
    username = validation["username"]
    normalized_url = validation["normalized_url"]

    # Cross-check user-supplied platform hint with detected platform
    if platform_hint and platform_hint not in (platform, "twitter"):
        return jsonify({
            "success": False,
            "error": (
                f"The URL belongs to {platform.capitalize()}, "
                f"but platform '{platform_hint}' was selected. "
                "Please select the correct platform."
            ),
            "step": "url_validation"
        }), 400

    # ------------------------------------------------------------------ #
    # Step 2: Call platform-specific extractor
    # ------------------------------------------------------------------ #
    extractors = {
        "instagram": extract_instagram,
        "facebook":  extract_facebook,
        "x":         extract_x,
    }
    extractor_fn = extractors[platform]
    extracted = extractor_fn(username, normalized_url)

    # ------------------------------------------------------------------ #
    # Step 3: Process features
    # ------------------------------------------------------------------ #
    proc = process_features(extracted)

    if proc["insufficient_data"]:
        # Return informative response — do not force a prediction
        return jsonify({
            "success":           True,
            "prediction":        None,
            "insufficient_data": True,
            "platform":          platform,
            "platform_label":    _platform_label(platform),
            "username":          username,
            "profile_url":       normalized_url,
            "extraction_method": extracted.get("extraction_method"),
            "extracted_details": _safe_extracted_details(extracted),
            "missing_fields":    proc.get("missing_features", []),
            "estimated_fields":  proc.get("estimated_features", []),
            "notes":             extracted.get("notes", []),
            "extraction_error":  extracted.get("error"),
            "message": (
                "Insufficient data was available to generate a prediction. "
                "The key account metrics (followers, following, posts) could not be "
                "retrieved from this platform. This is typically because the platform's "
                "API requires authentication or the account is private."
            ),
            "disclaimer": _disclaimer(),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }), 200

    # ------------------------------------------------------------------ #
    # Step 4: Run ML prediction
    # ------------------------------------------------------------------ #
    if not predictor or not predictor.model:
        return jsonify({
            "success": False,
            "error": "Machine learning model is not loaded. Train the model first."
        }), 503

    ml_result = predictor.predict(proc["feature_dict"])

    if not ml_result.get("success"):
        return jsonify({
            "success": False,
            "error":   "ML prediction failed.",
            "details": ml_result.get("errors", []),
        }), 500

    # ------------------------------------------------------------------ #
    # Step 5: Compose and return full response
    # ------------------------------------------------------------------ #
    return jsonify({
        "success":           True,
        "insufficient_data": False,

        # Platform & account info
        "platform":          platform,
        "platform_label":    _platform_label(platform),
        "username":          username,
        "profile_url":       normalized_url,
        "extraction_method": extracted.get("extraction_method"),

        # Extracted account details (for display)
        "extracted_details": _safe_extracted_details(extracted),
        "missing_fields":    proc.get("missing_features", []),
        "estimated_fields":  proc.get("estimated_features", []),
        "notes":             extracted.get("notes", []),
        "extraction_error":  extracted.get("error"),

        # ML results
        "prediction":        ml_result["prediction"],
        "risk_score":        ml_result["risk_score"],
        "risk_level":        ml_result["risk_level"],
        "status_theme":      ml_result["status_theme"],
        "genuine_probability": ml_result["genuine_probability"],
        "fake_probability":    ml_result["fake_probability"],
        "explanation":         ml_result["explanation"],
        "submitted_features":  ml_result["submitted_features"],
        "model_metadata":      ml_result["model_metadata"],

        "disclaimer": _disclaimer(),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }), 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _platform_label(platform: str) -> str:
    return {"instagram": "Instagram", "facebook": "Facebook", "x": "X (Twitter)"}.get(platform, platform)


def _safe_extracted_details(extracted: dict) -> dict:
    """Returns only the data fields (not internal bookkeeping keys)."""
    return {
        k: extracted.get(k)
        for k in [
            "followers_count", "following_count", "posts_count",
            "has_profile_pic", "has_bio", "has_external_link",
            "account_age_bracket", "is_private",
        ]
    }


def _disclaimer() -> str:
    return (
        "Educational Prototype Notice: This prediction is an algorithmic estimate based "
        "on publicly available profile metadata. It cannot conclusively confirm or disprove "
        "account authenticity. API data availability varies by platform and access level."
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Fake Account Detection Backend on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)

