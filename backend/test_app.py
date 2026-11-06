"""
Automated Test Suite for Fake Social Media Account Detection System
--------------------------------------------------------------------
Updated to test the simplified 8-feature schema:
  - followers_count
  - following_count
  - posts_count
  - account_age_bracket
  - has_bio
  - has_external_link
  - has_profile_pic
  - account_type (optional)
"""

import os
import sys
import json
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app import app

class FakeAccountDetectionTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_01_health_endpoint(self):
        """Verify GET /api/health returns 200 and healthy status."""
        response = self.app.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertIn("version", data)

    def test_02_metrics_endpoint(self):
        """Verify GET /api/metrics returns accurate training metrics and importances."""
        response = self.app.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        metrics = data["metrics"]
        self.assertIn("accuracy", metrics)
        self.assertIn("feature_importances", metrics)
        self.assertGreater(len(metrics["feature_importances"]), 0)

    def test_03_predict_genuine_account(self):
        """Verify POST /api/predict correctly classifies organic human account."""
        payload = {
            "followers_count": 650,
            "following_count": 480,
            "posts_count": 180,
            "account_age_bracket": "1_to_2_years",
            "has_bio": 1,
            "has_external_link": 0,
            "has_profile_pic": 1,
            "account_type": "private"
        }
        response = self.app.post("/api/predict", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["prediction"], "Likely Genuine")
        self.assertLess(data["risk_score"], 50.0)
        self.assertGreater(len(data["explanation"]["trust_factors"]), 0)

    def test_04_predict_spam_bot_account(self):
        """Verify POST /api/predict correctly flags mass-follower spam bot."""
        payload = {
            "followers_count": 12,
            "following_count": 4200,
            "posts_count": 1,
            "account_age_bracket": "lt_1_month",
            "has_bio": 0,
            "has_external_link": 0,
            "has_profile_pic": 0,
            "account_type": "public"
        }
        response = self.app.post("/api/predict", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["prediction"], "Suspicious")
        self.assertGreaterEqual(data["risk_score"], 50.0)
        self.assertGreater(len(data["explanation"]["risk_factors"]), 0)

    def test_05_predict_missing_fields_validation(self):
        """Verify POST /api/predict returns 400 when fields are omitted."""
        payload = {"followers_count": 100}
        response = self.app.post("/api/predict", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("details", data)

    def test_06_predict_negative_values_validation(self):
        """Verify POST /api/predict returns 400 when physical values are negative."""
        payload = {
            "followers_count": -10,
            "following_count": 200,
            "posts_count": -5,
            "account_age_bracket": "1_to_6_months",
            "has_bio": 1,
            "has_external_link": 0,
            "has_profile_pic": 1
        }
        response = self.app.post("/api/predict", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])

    def test_07_frontend_serving(self):
        """Verify frontend index.html, style.css, and script.js are served."""
        r_index = self.app.get("/")
        self.assertEqual(r_index.status_code, 200)
        self.assertIn(b"Fake Account Detector AI", r_index.data)

        r_css = self.app.get("/style.css")
        self.assertEqual(r_css.status_code, 200)

        r_js = self.app.get("/script.js")
        self.assertEqual(r_js.status_code, 200)

if __name__ == "__main__":
    unittest.main()
