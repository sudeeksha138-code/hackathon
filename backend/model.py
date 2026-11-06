"""
Model Inference & Explainability Engine
---------------------------------------
Loads the trained Random Forest Classifier and evaluates new account profiles.
Updated for simplified user-friendly schema:
  - followers_count (int >= 0)
  - following_count (int >= 0)
  - posts_count (int >= 0)
  - account_age_bracket (0: <1 mo, 1: 1-6 mo, 2: 6-12 mo, 3: 1-2 yr, 4: >2 yr)
  - has_bio (0 or 1)
  - has_external_link (0 or 1)
  - has_profile_pic (0 or 1)
  - is_private (0 or 1, optional, defaults to 0)
"""

import os
import json
import joblib
import numpy as np

FEATURE_NAMES = [
    "followers_count",
    "following_count",
    "posts_count",
    "account_age_bracket",
    "has_bio",
    "has_external_link",
    "has_profile_pic",
    "is_private"
]

AGE_BRACKET_MAP = {
    0: 0, "0": 0, "lt_1_month": 0, "less than 1 month": 0, "< 1 month": 0,
    1: 1, "1": 1, "1_to_6_months": 1, "1–6 months": 1, "1-6 months": 1,
    2: 2, "2": 2, "6_to_12_months": 2, "6–12 months": 2, "6-12 months": 2,
    3: 3, "3": 3, "1_to_2_years": 3, "1–2 years": 3, "1-2 years": 3,
    4: 4, "4": 4, "gt_2_years": 4, "more than 2 years": 4, "> 2 years": 4
}

AGE_BRACKET_LABELS = {
    0: "Less than 1 month",
    1: "1–6 months",
    2: "6–12 months",
    3: "1–2 years",
    4: "More than 2 years"
}

class FakeAccountPredictor:
    def __init__(self, model_path: str = None, metrics_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "fake_account_model.pkl")
        if metrics_path is None:
            metrics_path = os.path.join(base_dir, "models", "model_metrics.json")
            
        self.model_path = model_path
        self.metrics_path = metrics_path
        self.model = None
        self.metadata = {}
        self.metrics = {}
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please run 'python backend/train_model.py' first."
            )
        
        payload = joblib.load(self.model_path)
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
            self.metadata = {k: v for k, v in payload.items() if k != "model"}
        else:
            self.model = payload
            self.metadata = {}

        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
            except Exception:
                self.metrics = {}

    def validate_inputs(self, data: dict) -> tuple[dict, list[str]]:
        """
        Validates incoming data dictionary for the simplified feature schema.
        Returns (sanitized_data_dict, list_of_error_strings).
        """
        errors = []
        sanitized = {}

        # 1. Numeric count fields: followers_count, following_count, posts_count
        count_fields = ["followers_count", "following_count", "posts_count"]
        for feat in count_fields:
            if feat not in data or data[feat] is None or data[feat] == "":
                errors.append(f"Missing required field: '{feat}'.")
                continue
            try:
                val = float(data[feat])
                if val < 0:
                    errors.append(f"Field '{feat}' cannot be negative (got {val}).")
                else:
                    sanitized[feat] = int(val)
            except (ValueError, TypeError):
                errors.append(f"Field '{feat}' must be a valid non-negative number.")

        # 2. Account Age Dropdown
        age_raw = data.get("account_age_bracket", data.get("account_age"))
        if age_raw is None or age_raw == "":
            errors.append("Please select an Account Age option from the dropdown.")
        else:
            # Normalize string for lookup
            lookup_key = str(age_raw).strip().lower()
            if lookup_key in AGE_BRACKET_MAP:
                sanitized["account_age_bracket"] = AGE_BRACKET_MAP[lookup_key]
            else:
                try:
                    int_val = int(age_raw)
                    if 0 <= int_val <= 4:
                        sanitized["account_age_bracket"] = int_val
                    else:
                        errors.append("Account Age must be between 0 and 4.")
                except (ValueError, TypeError):
                    errors.append(f"Invalid Account Age selection: '{age_raw}'.")

        # 3. Binary Yes/No toggles
        binary_fields = [
            ("has_bio", "Does the account have a bio?"),
            ("has_external_link", "Does the bio contain an external link?"),
            ("has_profile_pic", "Profile picture available?")
        ]
        for key, label in binary_fields:
            if key not in data or data[key] is None:
                errors.append(f"Please select Yes or No for '{label}'.")
                continue
            raw_val = data[key]
            if str(raw_val).strip().lower() in ["1", "true", "yes"]:
                sanitized[key] = 1
            elif str(raw_val).strip().lower() in ["0", "false", "no"]:
                sanitized[key] = 0
            else:
                errors.append(f"Field '{key}' must be binary Yes (1) or No (0).")

        # Consistency check: if has_bio is 0, bio cannot have external link
        if sanitized.get("has_bio") == 0:
            sanitized["has_external_link"] = 0

        # 4. Optional Account Type (Public / Private)
        account_type_raw = data.get("account_type", data.get("is_private", "public"))
        if str(account_type_raw).strip().lower() in ["private", "1", "true"]:
            sanitized["is_private"] = 1
        else:
            sanitized["is_private"] = 0

        return sanitized, errors

    def generate_explanations(self, data: dict, fake_prob: float) -> dict:
        """
        Generates intuitive risk and trust factor explanations based on user-friendly fields.
        """
        risk_factors = []
        trust_factors = []
        
        followers = data["followers_count"]
        following = data["following_count"]
        posts = data["posts_count"]
        age_bracket = data["account_age_bracket"]
        has_bio = data["has_bio"]
        has_link = data["has_external_link"]
        has_pic = data["has_profile_pic"]
        is_private = data["is_private"]

        ratio = followers / (following + 1.0)
        age_label = AGE_BRACKET_LABELS.get(age_bracket, "Unknown")

        # 1. Profile Picture
        if has_pic == 0:
            risk_factors.append({
                "factor": "Default Profile Picture",
                "severity": "high",
                "detail": "Account lacks a custom profile photo. Automated bot farms frequently leave avatars as platform defaults."
            })
        else:
            trust_factors.append({
                "factor": "Custom Profile Picture",
                "detail": "Avatar photo uploaded, consistent with genuine human user profiles."
            })

        # 2. Follower-to-Following Ratio
        if following > 400 and ratio < 0.05:
            risk_factors.append({
                "factor": "Severe Follower Asymmetry",
                "severity": "high",
                "detail": f"Account follows {following} users but only has {followers} followers (ratio: {ratio:.3f}). Aggressive mass-following is typical of automated bots."
            })
        elif ratio >= 0.5:
            trust_factors.append({
                "factor": "Balanced Follower Ratio",
                "detail": f"Follower-to-following ratio is balanced ({ratio:.2f}), reflecting organic social network connections."
            })

        # 3. Account Age Bracket
        if age_bracket == 0:
            risk_factors.append({
                "factor": "Brand New Account (< 1 month)",
                "severity": "medium",
                "detail": "Account was created less than 1 month ago. Malicious and throwaway spam accounts are predominantly young."
            })
        elif age_bracket in [3, 4]:
            trust_factors.append({
                "factor": f"Established Account ({age_label})",
                "detail": f"Account tenure is well established ({age_label}). Longevity significantly reduces probability of throwaway botting."
            })

        # 4. External Link & Bio
        if has_link == 1 and age_bracket <= 1:
            risk_factors.append({
                "factor": "External Link on Young Account",
                "severity": "high",
                "detail": "Young account contains an external link in bio, a pattern frequently observed in phishing or affiliate scams."
            })
        elif has_link == 1:
            trust_factors.append({
                "factor": "External Link Present",
                "detail": "Profile links to an external website or portfolio."
            })

        if has_bio == 0:
            risk_factors.append({
                "factor": "No Biography",
                "severity": "low",
                "detail": "No bio text provided. Common in inactive, scraper, or unconfigured bot accounts."
            })
        else:
            trust_factors.append({
                "factor": "Profile Biography Present",
                "detail": "Account contains a populated biography section."
            })

        # 5. Posts Activity
        if posts == 0 and following > 80:
            risk_factors.append({
                "factor": "Zero Posts with Active Following",
                "severity": "medium",
                "detail": f"Zero posts published despite actively following {following} accounts."
            })
        elif posts >= 10:
            trust_factors.append({
                "factor": "Active Post History",
                "detail": f"Account has published {posts} posts, indicating genuine social participation."
            })

        # 6. Privacy Setting
        if is_private == 1:
            trust_factors.append({
                "factor": "Private Profile Setting",
                "detail": "Account privacy is set to Private. Mass spam and promo bots almost exclusively operate public accounts for reach."
            })

        # Summary text
        if fake_prob >= 0.70:
            summary = (
                "The machine learning model detected strong behavioral anomalies, including severe follow asymmetry, "
                "missing profile assets, or new account tenure characteristic of synthetic spam accounts."
            )
        elif fake_prob >= 0.50:
            summary = (
                "The account exhibits moderate risk indicators. While not conclusive proof of a bot, several features "
                "(such as low tenure or follower asymmetry) deviate from typical genuine user baselines."
            )
        elif fake_prob >= 0.25:
            summary = (
                "The account appears largely genuine with minor irregularities. The risk score remains within safe thresholds."
            )
        else:
            summary = (
                "The account features closely align with legitimate, organically grown user profiles with strong trust indicators."
            )

        return {
            "summary": summary,
            "risk_factors": risk_factors,
            "trust_factors": trust_factors
        }

    def predict(self, raw_data: dict) -> dict:
        """
        Executes full validation, inference, and explanation pipeline.
        """
        sanitized_data, errors = self.validate_inputs(raw_data)
        if errors:
            return {
                "success": False,
                "errors": errors
            }

        # Format feature vector in exact feature order
        feature_vector = np.array([[sanitized_data[col] for col in FEATURE_NAMES]], dtype=np.float32)

        # Scikit-learn inference
        probabilities = self.model.predict_proba(feature_vector)[0]
        genuine_prob = float(probabilities[0])
        fake_prob = float(probabilities[1])

        risk_score = round(fake_prob * 100, 1)

        if risk_score >= 50.0:
            prediction_label = "Suspicious"
            status_theme = "danger" if risk_score >= 75.0 else "warning"
        else:
            prediction_label = "Likely Genuine"
            status_theme = "success"

        if risk_score < 25.0:
            risk_level = "Low Risk"
        elif risk_score < 50.0:
            risk_level = "Moderate Risk"
        elif risk_score < 75.0:
            risk_level = "High Risk"
        else:
            risk_level = "Critical Risk"

        explanations = self.generate_explanations(sanitized_data, fake_prob)

        # Build user-friendly submitted features representation
        friendly_features = dict(sanitized_data)
        friendly_features["account_age_label"] = AGE_BRACKET_LABELS.get(sanitized_data["account_age_bracket"], "Unknown")
        friendly_features["account_type_label"] = "Private" if sanitized_data["is_private"] == 1 else "Public"

        return {
            "success": True,
            "prediction": prediction_label,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "status_theme": status_theme,
            "genuine_probability": round(genuine_prob * 100, 1),
            "fake_probability": round(fake_prob * 100, 1),
            "submitted_features": friendly_features,
            "explanation": explanations,
            "model_metadata": {
                "algorithm": self.metrics.get("algorithm", "Random Forest Classifier"),
                "dataset_type": self.metrics.get("dataset_type", "synthetic_demonstration_dataset"),
                "is_demonstration": self.metrics.get("is_demonstration", True),
                "model_accuracy": self.metrics.get("accuracy", 1.0)
            },
            "disclaimer": (
                "Educational Prototype Notice: This prediction is an algorithmic estimate based on statistical feature "
                "patterns. It cannot conclusively confirm or disprove authentic account identity."
            )
        }
