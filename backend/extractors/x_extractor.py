"""
X (Twitter) Account Data Extractor
-------------------------------------
Retrieves publicly available X/Twitter profile information via the
official X API v2.

API Access Reality (as of 2024):
  - X API v2 Free tier supports the GET /2/users/by/username/:username
    endpoint for public user lookups, including public_metrics.
  - A Bearer Token (App-Only Auth) is sufficient — no user login required.
  - public_metrics includes: followers_count, following_count, tweet_count,
    listed_count.
  - User object includes: description (bio), created_at, profile_image_url,
    entities (for external URLs in bio), protected (is_private).

This is the most data-rich extraction path in this application because
the X API v2 Free tier genuinely supports public user lookups.

To use: set X_BEARER_TOKEN in your .env file or environment variables.
"""

from __future__ import annotations

import os
import json
import logging
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

X_API_BASE = "https://api.twitter.com/2"
REQUEST_TIMEOUT = 10


def extract_x(username: str, profile_url: str) -> dict:
    """
    Extracts public X/Twitter account features for `username`.

    Returns the standard extractor dict (see instagram_extractor.py for schema).
    """
    result = _base_result(username, profile_url)

    bearer_token = os.environ.get("X_BEARER_TOKEN", "").strip()

    if not bearer_token:
        result["notes"].append(
            "No X_BEARER_TOKEN environment variable found. "
            "X API v2 requires a Bearer Token from a registered developer app. "
            "Free tier (Essential access) is sufficient for public user lookups."
        )
        result["notes"].append(
            "To enable X data extraction: register at developer.twitter.com, "
            "create an app, copy the Bearer Token, and set X_BEARER_TOKEN in your .env file."
        )
        result["extraction_method"] = "none — bearer token not configured"
        result["missing_fields"] = _all_data_fields()
        return result

    try:
        api_data = _x_api_user_lookup(username, bearer_token)
        return _map_api_response(api_data, result)
    except _ApiError as exc:
        result["error"] = f"X API error: {exc}"
        result["extraction_method"] = "x_api_v2 — failed"
        result["missing_fields"] = _all_data_fields()
        return result
    except Exception as exc:
        logger.exception("Unexpected error during X extraction")
        result["error"] = f"Unexpected error during X data extraction: {exc}"
        result["extraction_method"] = "x_api_v2 — exception"
        result["missing_fields"] = _all_data_fields()
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _x_api_user_lookup(username: str, token: str) -> dict:
    """Calls X API v2 to look up a public user by username."""
    params = urllib.parse.urlencode({
        "user.fields": (
            "public_metrics,created_at,description,"
            "profile_image_url,entities,protected"
        ),
    })
    url = f"{X_API_BASE}/users/by/username/{urllib.parse.quote(username)}?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "FakeAccountDetectorAI/1.0 (Educational Prototype)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise _ApiError(f"HTTP {exc.code}: {body[:300]}") from exc
    except urllib.error.URLError as exc:
        raise _ApiError(f"Network error: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise _ApiError(f"Invalid JSON response: {exc}") from exc


def _map_api_response(api_response: dict, result: dict) -> dict:
    """Maps X API v2 response to our standard schema."""
    result["extraction_method"] = "x_api_v2"

    # Check for API-level errors
    if "errors" in api_response and "data" not in api_response:
        errors = api_response["errors"]
        raise _ApiError(f"X API returned errors: {errors}")

    user = api_response.get("data", {})
    if not user:
        raise _ApiError("X API returned an empty user object.")

    metrics = user.get("public_metrics", {})
    entities = user.get("entities", {})
    description = user.get("description", "")
    created_at = user.get("created_at", "")
    pic_url = user.get("profile_image_url", "")
    is_protected = user.get("protected", False)

    # --- Follower / Following / Posts counts ---
    followers = metrics.get("followers_count")
    following = metrics.get("following_count")
    tweets    = metrics.get("tweet_count")

    if followers is not None:
        result["followers_count"] = int(followers)
    else:
        result["missing_fields"].append("followers_count")

    if following is not None:
        result["following_count"] = int(following)
    else:
        result["missing_fields"].append("following_count")

    if tweets is not None:
        result["posts_count"] = int(tweets)
    else:
        result["missing_fields"].append("posts_count")

    # --- has_bio ---
    result["has_bio"] = 1 if (description and description.strip()) else 0

    # --- has_external_link ---
    # Entities.url is populated when the user has a URL in their bio/profile
    url_entities = entities.get("url", {}).get("urls", [])
    desc_entities = entities.get("description", {}).get("urls", [])
    has_link = bool(url_entities or desc_entities)
    result["has_external_link"] = 1 if has_link else 0

    # --- has_profile_pic ---
    # X always returns a profile_image_url; default is typically "_normal.jpg"
    # containing "default_profile" in the URL
    if pic_url:
        result["has_profile_pic"] = 0 if "default_profile" in pic_url else 1
    else:
        result["has_profile_pic"] = None
        result["missing_fields"].append("has_profile_pic")

    # --- account_age_bracket ---
    if created_at:
        result["account_age_bracket"] = _created_at_to_bracket(created_at)
    else:
        result["missing_fields"].append("account_age_bracket")

    # --- is_private ---
    result["is_private"] = 1 if is_protected else 0

    result["notes"].append(
        "Data extracted via X API v2 (public_metrics). "
        "All values reflect publicly available profile information."
    )

    return result


def _created_at_to_bracket(created_at_str: str) -> int:
    """
    Converts X API's ISO 8601 created_at string to our age bracket encoding.
    Returns 0–4 or None on parse failure.

    Brackets:
      0: < 1 month
      1: 1–6 months
      2: 6–12 months
      3: 1–2 years
      4: > 2 years
    """
    try:
        # X API returns: "2020-01-29T00:00:00.000Z"
        created = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        days_old = (now - created).days

        if days_old < 30:
            return 0
        elif days_old < 180:
            return 1
        elif days_old < 365:
            return 2
        elif days_old < 730:
            return 3
        else:
            return 4
    except Exception:
        return None


def _base_result(username: str, profile_url: str) -> dict:
    return {
        "success":           True,
        "username":          username,
        "profile_url":       profile_url,
        "platform":          "x",
        "extraction_method": "pending",
        "followers_count":   None,
        "following_count":   None,
        "posts_count":       None,
        "has_profile_pic":   None,
        "has_bio":           None,
        "has_external_link": None,
        "account_age_bracket": None,
        "is_private":        None,
        "missing_fields":    [],
        "notes":             [],
        "error":             None,
    }


def _all_data_fields() -> list[str]:
    return [
        "followers_count", "following_count", "posts_count",
        "has_profile_pic", "has_bio", "has_external_link",
        "account_age_bracket", "is_private",
    ]


class _ApiError(Exception):
    pass

