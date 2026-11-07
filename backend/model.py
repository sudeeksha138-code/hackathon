"""
Model Inference & Explainability Engine
---------------------------------------
Loads the trained Random Forest Classifier and evaluates social media account authenticity.
Uses an ensemble of Random Forest inference and multi-signal forensic risk calibration:
  Raw user inputs:
    - followers_count (int >= 0)
    - following_count (int >= 0)
    - posts_count (int >= 0)
    - account_age_bracket (0: <1 mo, 1: 1-6 mo, 2: 6-12 mo, 3: 1-2 yr, 4: >2 yr)
    - has_bio (0 or 1)
    - has_external_link (0 or 1)
    - has_profile_pic (0 or 1)
    - is_private (0 or 1)
  Engineered features:
    - follower_following_ratio: followers / (following + 1)
    - follow_asymmetry: (following - followers) / (following + followers + 1)
    - post_rate_per_age: posts / (account_age_bracket + 1)
"""

import os
import json
import joblib
import numpy as np

RAW_FEATURE_NAMES = [
    "followers_count",
    "following_count",
    "posts_count",
    "account_age_bracket",
    "has_bio",
    "has_external_link",
    "has_profile_pic",
    "is_private"
]

ENGINEERED_FEATURE_NAMES = [
    "follower_following_ratio",
    "follow_asymmetry",
    "post_rate_per_age"
]

FEATURE_NAMES = RAW_FEATURE_NAMES + ENGINEERED_FEATURE_NAMES

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

        # 4. Optional Account Type (Public / Private)
        account_type_raw = data.get("account_type", data.get("is_private", "public"))
        if str(account_type_raw).strip().lower() in ["private", "1", "true"]:
            sanitized["is_private"] = 1
        else:
            sanitized["is_private"] = 0

        # Compute engineered features
        if not errors:
            followers = float(sanitized["followers_count"])
            following = float(sanitized["following_count"])
            posts = float(sanitized["posts_count"])
            age = float(sanitized["account_age_bracket"])

            sanitized["follower_following_ratio"] = float(followers / (following + 1.0))
            sanitized["follow_asymmetry"] = float((following - followers) / (following + followers + 1.0))
            sanitized["post_rate_per_age"] = float(posts / (age + 1.0))

        return sanitized, errors

    def generate_explanations(self, data: dict, final_risk: float) -> dict:
        """
        Generates intuitive, realistic risk and trust factor explanations.
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

        # 2. Follower-to-Following Ratio & Asymmetry
        if following > 400 and ratio < 0.05:
            risk_factors.append({
                "factor": "Severe Follower Asymmetry",
                "severity": "high",
                "detail": f"Account follows {following:,} users but only has {followers:,} followers (ratio: {ratio:.3f}). Aggressive mass-following is typical of automated bots."
            })
        elif following > 200 and ratio < 0.15:
            risk_factors.append({
                "factor": "Follower Disparity",
                "severity": "medium",
                "detail": f"Account follows significantly more profiles ({following:,}) than follow it back ({followers:,}, ratio: {ratio:.2f})."
            })
        elif ratio >= 0.4 and ratio <= 25.0:
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
                "detail": "Young account contains an external link, a pattern frequently observed in phishing, crypto, or affiliate scams."
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

        # 5. Posts Activity & Tempo
        if posts == 0 and following > 80:
            risk_factors.append({
                "factor": "Zero Posts with Active Following",
                "severity": "medium",
                "detail": f"Zero posts published despite actively following {following:,} accounts."
            })
        elif posts > 60 and age_bracket == 0 and ratio < 0.15:
            risk_factors.append({
                "factor": "Anomalous High-Frequency Posting",
                "severity": "high",
                "detail": f"Account published {posts:,} posts in less than 1 month (>2-5 posts/day) with low engagement, characteristic of automated scrapers or reposting bots."
            })
        elif posts >= 5 and age_bracket >= 1:
            trust_factors.append({
                "factor": "Active Post History",
                "detail": f"Account has published {posts:,} posts over established tenure, indicating genuine social participation."
            })

        # 6. Privacy Setting
        if is_private == 1:
            if following > 400 and ratio < 0.05:
                risk_factors.append({
                    "factor": "Private Follow-Trap Pattern",
                    "severity": "medium",
                    "detail": f"Account is private while aggressively following {following:,} users with only {followers:,} followers. Scraping and follow-bait bots often lock profiles to evade platform detection."
                })
            elif ratio >= 0.3:
                trust_factors.append({
                    "factor": "Private Profile Setting",
                    "detail": "Account privacy is set to Private with organic follower balance, consistent with personal human privacy protection."
                })

        # Summary text
        if final_risk >= 75.0:
            summary = (
                "The forensic AI detected critical bot indicators, including severe follow asymmetry, "
                "missing avatar assets, and anomalous activity patterns characteristic of automated spam or scraper accounts."
            )
        elif final_risk >= 50.0:
            summary = (
                "The account exhibits moderate-to-high risk indicators. Several parameters "
                "(such as young tenure, follower asymmetry, or missing profile data) deviate substantially from authentic user baselines."
            )
        elif final_risk >= 25.0:
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

    def compute_forensic_risk_score(self, sanitized_data: dict, rf_fake_prob: float) -> float:
        """
        Ensemble Forensic Synthesizer:
        Combines Random Forest probability with rule-guided anomaly penalties
        to prevent edge-case misclassifications (e.g. 4200 following with 10 followers).
        """
        followers = sanitized_data["followers_count"]
        following = sanitized_data["following_count"]
        posts = sanitized_data["posts_count"]
        age_bracket = sanitized_data["account_age_bracket"]
        has_bio = sanitized_data["has_bio"]
        has_link = sanitized_data["has_external_link"]
        has_pic = sanitized_data["has_profile_pic"]
        is_private = sanitized_data["is_private"]
        ratio = sanitized_data["follower_following_ratio"]

        # Calculate empirical anomaly penalties
        anomaly_score = 0.0

        # Fatal Flag 1: Severe follow asymmetry (following hundreds/thousands with few followers)
        if following >= 400 and ratio < 0.02:
            anomaly_score += 0.48
        elif following >= 200 and ratio < 0.06:
            anomaly_score += 0.35
        elif following >= 100 and ratio < 0.12:
            anomaly_score += 0.20

        # Fatal Flag 2: Default avatar
        if has_pic == 0:
            anomaly_score += 0.22 if following > 50 else 0.15

        # Fatal Flag 3: Brand new account (<1 month)
        if age_bracket == 0:
            anomaly_score += 0.14
        elif age_bracket == 1:
            anomaly_score += 0.06

        # Fatal Flag 4: No bio
        if has_bio == 0:
            anomaly_score += 0.10

        # Fatal Flag 5: External link on young account
        if has_link == 1 and age_bracket <= 1:
            anomaly_score += 0.15

        # Fatal Flag 6: Abnormal post tempo (e.g. >50 posts in < 1 month with low followers)
        if posts > 50 and age_bracket == 0 and ratio < 0.15:
            anomaly_score += 0.18

        # Genuine mitigations (reduce false positives on authentic users)
        mitigation = 0.0
        if ratio >= 0.5 and has_pic == 1 and age_bracket >= 2:
            mitigation += 0.35
        if ratio >= 0.8 and has_bio == 1 and has_pic == 1:
            mitigation += 0.25
        if age_bracket >= 3 and has_pic == 1:
            mitigation += 0.20

        net_heuristic = max(0.0, min(0.98, anomaly_score - mitigation))

        # If critical red flags are present (e.g. following > 400 with ratio < 0.02, no pic, young age),
        # enforce a strict risk floor so bots cannot fool the classifier
        if following >= 400 and ratio < 0.03 and (has_pic == 0 or age_bracket <= 1):
            risk_floor = 0.85
        elif following >= 200 and ratio < 0.05 and has_pic == 0:
            risk_floor = 0.75
        elif following > 80 and posts == 0 and has_pic == 0 and has_bio == 0:
            risk_floor = 0.65
        else:
            risk_floor = 0.0

        # Blended ensemble: 60% ML Model + 40% Forensic Heuristics, bounded by risk_floor
        blended_prob = 0.60 * rf_fake_prob + 0.40 * net_heuristic
        final_fake_prob = max(blended_prob, risk_floor, rf_fake_prob if rf_fake_prob >= 0.75 else 0.0)

        # Ensure genuine accounts with solid trust signals stay safely in genuine tier
        if ratio >= 0.6 and has_pic == 1 and age_bracket >= 2 and following < 2000:
            final_fake_prob = min(final_fake_prob, 0.15)
        elif ratio >= 0.4 and has_pic == 1 and age_bracket >= 1 and following < 1000:
            final_fake_prob = min(final_fake_prob, 0.25)

        return float(np.clip(final_fake_prob, 0.01, 0.99))

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

        # Format feature vector in exact feature order (11 features)
        feature_vector = np.array([[sanitized_data[col] for col in FEATURE_NAMES]], dtype=np.float32)

        # Scikit-learn inference
        probabilities = self.model.predict_proba(feature_vector)[0]
        rf_genuine_prob = float(probabilities[0])
        rf_fake_prob = float(probabilities[1])

        # Synthesize with forensic calibration
        final_fake_prob = self.compute_forensic_risk_score(sanitized_data, rf_fake_prob)
        final_genuine_prob = 1.0 - final_fake_prob

        risk_score = round(final_fake_prob * 100, 1)

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

        explanations = self.generate_explanations(sanitized_data, risk_score)

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
            "genuine_probability": round(final_genuine_prob * 100, 1),
            "fake_probability": round(final_fake_prob * 100, 1),
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
