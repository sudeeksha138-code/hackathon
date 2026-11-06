"""
Flask Backend API for AI-Based Fake Social Media Account Detection System
--------------------------------------------------------------------------
Endpoints:
  GET  /              -> Serves the Frontend Web App
  GET  /api/health    -> API Health Check & System Status
  GET  /api/metrics   -> Returns Model Performance & Dataset Metrics
  POST /api/predict   -> Predicts Account Authenticity & Risk Score
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Fake Account Detection Backend on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)

