"""
Instagram Account Data Extractor
----------------------------------
Attempts to retrieve publicly available Instagram profile information.

API Access Reality (as of 2024):
  - Instagram Graph API requires an approved Meta developer app AND a
    user/page access token. Without approval, almost no profile data
    is accessible via the official API.
  - Instagram Basic Display API was deprecated (Sep 2024).
  - Instagram oEmbed API (api.instagram.com/oembed/) returns metadata
    only for public posts/reels, not for profiles.
  - Unauthenticated scraping of instagram.com is explicitly prohibited
    by Instagram's Terms of Service and is not implemented here.

What this extractor can do:
  - If INSTAGRAM_ACCESS_TOKEN environment variable is set and the token
    has the right permissions (instagram_basic / pages_read_engagement),
    it will attempt Graph API calls for page/business accounts.
  - Otherwise, it returns a transparent "no data available" response so
    the caller can decide how to proceed (manual form fallback, etc.).

We NEVER fabricate, guess, or randomly generate account statistics.
"""

from __future__ import annotations

import os
import json
import logging
import urllib.request
import urllib.parse
import urllib.error

logger = logging.getLogger(__name__)

GRAPH_API_BASE = "https://graph.facebook.com/v19.0"
REQUEST_TIMEOUT = 10  # seconds


def extract_instagram(username: str, profile_url: str) -> dict:
    """
    Attempts to extract public Instagram account features for `username`.

    Returns
    -------
    dict with keys:
      - success (bool)
      - username (str)
      - profile_url (str)
      - platform (str) = "instagram"
      - extraction_method (str)
      - followers_count (int|None)
      - following_count (int|None)
      - posts_count (int|None)
      - has_profile_pic (int|None)  — 0/1 or None
      - has_bio (int|None)
      - has_external_link (int|None)
      - account_age_bracket (int|None)
      - is_private (int|None)
      - missing_fields (list[str])
      - notes (list[str])
      - error (str|None)
    """
    result = _base_result(username, profile_url)

    access_token = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "").strip()

    if not access_token:
        result["notes"].append(
            "No INSTAGRAM_ACCESS_TOKEN environment variable found. "
            "Instagram Graph API requires an approved Meta developer app with a valid "
            "access token. Without it, no profile data can be retrieved via the official API."
        )
        result["notes"].append(
            "To enable Instagram data extraction: set INSTAGRAM_ACCESS_TOKEN in your "
            ".env file or environment. See README.md for setup instructions."
        )
        result["extraction_method"] = "none — api token not configured"
        result["missing_fields"] = _all_data_fields()
        return result

    # Try Graph API with token
    try:
        data = _graph_api_user_lookup(username, access_token)
        return _map_graph_response(data, result)
    except _ApiError as exc:
        result["error"] = f"Instagram Graph API error: {exc}"
        result["extraction_method"] = "graph_api — failed"
        result["missing_fields"] = _all_data_fields()
        return result
    except Exception as exc:
        logger.exception("Unexpected error during Instagram extraction")
        result["error"] = f"Unexpected error during Instagram data extraction: {exc}"
        result["extraction_method"] = "graph_api — exception"
        result["missing_fields"] = _all_data_fields()
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _graph_api_user_lookup(username: str, token: str) -> dict:
    """Calls Graph API to look up an Instagram Business/Creator account."""
    # Instagram Graph API works with IGID (numeric). We first search by username
    # via the pages endpoint (requires pages_read_engagement permission).
    params = urllib.parse.urlencode({
        "fields": "id,username,name,biography,followers_count,follows_count,"
                  "media_count,profile_picture_url,website",
        "access_token": token,
    })
    url = f"{GRAPH_API_BASE}/{urllib.parse.quote(username)}?{params}"
    return _http_get_json(url)


def _map_graph_response(api_data: dict, result: dict) -> dict:
    """Maps Graph API response fields to our standard schema."""
    result["extraction_method"] = "instagram_graph_api"

    followers = api_data.get("followers_count")
    following = api_data.get("follows_count")
    posts     = api_data.get("media_count")
    bio_text  = api_data.get("biography", "")
    pic_url   = api_data.get("profile_picture_url", "")
    website   = api_data.get("website", "")

    if followers is not None:
        result["followers_count"] = int(followers)
    else:
        result["missing_fields"].append("followers_count")

    if following is not None:
        result["following_count"] = int(following)
    else:
        result["missing_fields"].append("following_count")

    if posts is not None:
        result["posts_count"] = int(posts)
    else:
        result["missing_fields"].append("posts_count")

    # has_bio: True if biography string is non-empty
    result["has_bio"] = 1 if (bio_text and bio_text.strip()) else 0

    # has_external_link: True if website field is populated
    result["has_external_link"] = 1 if (website and website.strip()) else 0

    # has_profile_pic: True if profile_picture_url is present
    # (default IG profile pictures always return a URL — this cannot differentiate
    #  custom vs default via Graph API without computer vision)
    result["has_profile_pic"] = 1 if (pic_url and "default" not in pic_url.lower()) else None
    if result["has_profile_pic"] is None:
        result["missing_fields"].append("has_profile_pic")

    # account_age_bracket — not available via Graph API (no created_at for IG users)
    result["missing_fields"].append("account_age_bracket")

    # is_private — Graph API only returns data for public/business accounts
    # If we got here, it's likely a public business/creator account
    result["is_private"] = 0
    result["notes"].append(
        "Instagram Graph API only returns data for public Business/Creator accounts. "
        "Personal private profiles are not accessible via this API."
    )

    return result


def _http_get_json(url: str) -> dict:
    """Makes an HTTPS GET request and parses JSON response."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FakeAccountDetectorAI/1.0 (Educational Prototype)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise _ApiError(f"HTTP {exc.code}: {body[:200]}") from exc
    except urllib.error.URLError as exc:
        raise _ApiError(f"Network error: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise _ApiError(f"Invalid JSON response: {exc}") from exc


def _base_result(username: str, profile_url: str) -> dict:
    return {
        "success":          True,
        "username":         username,
        "profile_url":      profile_url,
        "platform":         "instagram",
        "extraction_method": "pending",
        "followers_count":  None,
        "following_count":  None,
        "posts_count":      None,
        "has_profile_pic":  None,
        "has_bio":          None,
        "has_external_link": None,
        "account_age_bracket": None,
        "is_private":       None,
        "missing_fields":   [],
        "notes":            [],
        "error":            None,
    }


def _all_data_fields() -> list[str]:
    return [
        "followers_count", "following_count", "posts_count",
        "has_profile_pic", "has_bio", "has_external_link",
        "account_age_bracket", "is_private",
    ]


class _ApiError(Exception):
    pass

