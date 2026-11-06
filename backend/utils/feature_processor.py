"""
Feature Processor
-----------------
Converts extractor output dictionaries into the 8-feature vector
expected by the Random Forest model.

Feature order (must match FEATURE_NAMES in model.py):
  0. followers_count
  1. following_count
  2. posts_count
  3. account_age_bracket  (0=<1mo, 1=1-6mo, 2=6-12mo, 3=1-2yr, 4=>2yr)
  4. has_bio              (0 or 1)
  5. has_external_link    (0 or 1)
  6. has_profile_pic      (0 or 1)
  7. is_private           (0 or 1)

Rules:
  - followers_count, following_count, posts_count are CORE features.
    If all three are None, the data is insufficient for a prediction.
  - account_age_bracket defaults to 2 (6-12 months) when unavailable.
  - has_bio, has_external_link, has_profile_pic default to 1 when unknown
    (conservative / least alarming assumption).
  - is_private defaults to 0 (public) when unknown.
  - Any defaulted feature is listed in estimated_features.
  - No value is ever fabricated — only safe neutral defaults are used,
    and the caller is informed of every default via estimated_features.
"""

from __future__ import annotations

FEATURE_NAMES = [
    "followers_count",
    "following_count",
    "posts_count",
    "account_age_bracket",
    "has_bio",
    "has_external_link",
    "has_profile_pic",
    "is_private",
]

# How many core numeric features (followers, following, posts) must be present
_CORE_COUNT_FEATURES = {"followers_count", "following_count", "posts_count"}

# Safe neutral defaults for non-critical features
_SAFE_DEFAULTS = {
    "account_age_bracket": 2,  # 6-12 months — middle ground
    "has_bio":             1,  # assume yes
    "has_external_link":   0,  # assume no
    "has_profile_pic":     1,  # assume yes
    "is_private":          0,  # assume public
}


def process_features(extracted: dict) -> dict:
    """
    Converts raw extractor output into a model-ready feature dictionary.

    Parameters
    ----------
    extracted : dict
        Output dict from an extractor. Expected keys (all optional):
          followers_count, following_count, posts_count,
          account_age_bracket, has_bio, has_external_link,
          has_profile_pic, is_private

    Returns
    -------
    dict with keys:
      - success (bool)
      - feature_dict (dict|None): feature_name → int value
      - missing_features (list[str]): features that could not be retrieved
      - estimated_features (list[str]): features that used safe defaults
      - insufficient_data (bool): True if too few core features are known
      - error (str|None)
    """
    missing_features: list[str] = []
    estimated_features: list[str] = []
    feature_dict: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # 1. Core count features — cannot be defaulted
    # ------------------------------------------------------------------ #
    core_available = 0
    for feat in _CORE_COUNT_FEATURES:
        raw = extracted.get(feat)
        if raw is None:
            missing_features.append(feat)
        else:
            try:
                val = int(float(raw))
                if val < 0:
                    val = 0
                feature_dict[feat] = val
                core_available += 1
            except (TypeError, ValueError):
                missing_features.append(feat)

    # If zero core features available → insufficient data
    if core_available == 0:
        return {
            "success":           False,
            "feature_dict":      None,
            "missing_features":  missing_features,
            "estimated_features": [],
            "insufficient_data": True,
            "error": (
                "Insufficient data: followers, following, and posts counts are all "
                "unavailable. A prediction cannot be made without at least one count metric."
            ),
        }

    # If some but not all core features are missing, fill with 0 and flag
    for feat in _CORE_COUNT_FEATURES:
        if feat not in feature_dict:
            feature_dict[feat] = 0
            estimated_features.append(feat)
            # Remove from missing (we're covering it with a default)
            if feat in missing_features:
                missing_features.remove(feat)

    # ------------------------------------------------------------------ #
    # 2. Non-critical features — use safe defaults when unavailable
    # ------------------------------------------------------------------ #
    non_critical = [
        "account_age_bracket",
        "has_bio",
        "has_external_link",
        "has_profile_pic",
        "is_private",
    ]
    for feat in non_critical:
        raw = extracted.get(feat)
        if raw is None:
            default = _SAFE_DEFAULTS[feat]
            feature_dict[feat] = default
            missing_features.append(feat)
            estimated_features.append(feat)
        else:
            try:
                # Binary booleans
                if feat in ("has_bio", "has_external_link", "has_profile_pic", "is_private"):
                    if str(raw).lower() in ("true", "yes", "1"):
                        feature_dict[feat] = 1
                    elif str(raw).lower() in ("false", "no", "0"):
                        feature_dict[feat] = 0
                    else:
                        feature_dict[feat] = int(bool(raw))
                else:
                    # account_age_bracket — integer 0-4
                    val = int(raw)
                    if 0 <= val <= 4:
                        feature_dict[feat] = val
                    else:
                        feature_dict[feat] = _SAFE_DEFAULTS[feat]
                        estimated_features.append(feat)
            except (TypeError, ValueError):
                feature_dict[feat] = _SAFE_DEFAULTS[feat]
                estimated_features.append(feat)

    # Consistency: if no bio, no external link possible
    if feature_dict.get("has_bio") == 0:
        feature_dict["has_external_link"] = 0

    return {
        "success":            True,
        "feature_dict":       feature_dict,
        "missing_features":   missing_features,
        "estimated_features": estimated_features,
        "insufficient_data":  False,
        "error":              None,
    }

