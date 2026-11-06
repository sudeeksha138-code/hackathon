"""
Facebook Account Data Extractor
---------------------------------
Attempts to retrieve publicly available Facebook profile/page information.

API Access Reality (as of 2024):
  - Facebook Graph API v19+ requires an approved developer app.
  - Personal profiles: virtually no data is accessible without the user's
    own access token (which requires them to log in and authorize the app).
  - Public Pages: limited data (name, fan_count / followers) is accessible
    via a Page Access Token or App Access Token, but only for Facebook Pages
    (not personal profiles).
  - The endpoint used here targets Facebook Pages (businesses, public figures)
    rather than personal user profiles.

What this extractor does:
  - If FACEBOOK_ACCESS_TOKEN is configured, attempts to look up the page
    with the given username/page-name via the Graph API.
  - For personal profiles, extraction will typically fail because the Graph
    API does not expose personal profile stats to third-party apps.
  - Returns a transparent no-data result when the token is not configured.

We NEVER fabricate, guess, or randomly generate statistics.
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
REQUEST_TIMEOUT = 10


def extract_facebook(username: str, profile_url: str) -> dict:
    """
    Attempts to extract public Facebook page/profile features for `username`.

    Returns the standard extractor dict (see instagram_extractor.py for schema).
    """
    result = _base_result(username, profile_url)

    access_token = os.environ.get("FACEBOOK_ACCESS_TOKEN", "").strip()

    if not access_token:
        result["notes"].append(
            "No FACEBOOK_ACCESS_TOKEN environment variable found. "
            "Facebook Graph API requires a valid Page Access Token or App Access Token. "
            "Without it, no profile data can be retrieved."
        )
        result["notes"].append(
            "Note: Facebook Graph API can only retrieve data for public Facebook Pages, "
            "not personal user profiles. Personal profile stats are never publicly accessible "
            "via the official API."
        )
        result["extraction_method"] = "none — api token not configured"
        result["missing_fields"] = _all_data_fields()
        return result

    try:
        data = _graph_api_page_lookup(username, access_token)
        return _map_graph_response(data, result)
    except _ApiError as exc:
        # Check if it's a 'not a page' or permissions error
        result["error"] = f"Facebook Graph API error: {exc}"
        result["notes"].append(
            "Facebook Graph API data access is restricted to public Facebook Pages. "
            "Personal profiles cannot be queried via the Graph API without the user's "
            "explicit authorization."
        )
        result["extraction_method"] = "graph_api — failed"
        result["missing_fields"] = _all_data_fields()
        return result
    except Exception as exc:
        logger.exception("Unexpected error during Facebook extraction")
        result["error"] = f"Unexpected error during Facebook data extraction: {exc}"
        result["extraction_method"] = "graph_api — exception"
        result["missing_fields"] = _all_data_fields()
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _graph_api_page_lookup(identifier: str, token: str) -> dict:
    """Looks up a Facebook Page by page name/ID via Graph API."""
    params = urllib.parse.urlencode({
        "fields": "id,name,username,biography,fan_count,followers_count,"
                  "posts.limit(1){id},picture,website,category",
        "access_token": token,
    })
    url = f"{GRAPH_API_BASE}/{urllib.parse.quote(identifier)}?{params}"
    return _http_get_json(url)


def _map_graph_response(api_data: dict, result: dict) -> dict:
    """Maps Graph API response fields to our standard schema."""
    result["extraction_method"] = "facebook_graph_api"

    # Facebook Pages expose fan_count or followers_count
    fan_count       = api_data.get("fan_count")
    followers_count = api_data.get("followers_count")
    bio_text        = api_data.get("biography", "")
    website         = api_data.get("website", "")
    posts_data      = api_data.get("posts", {})
    picture_data    = api_data.get("picture", {})

    # Use fan_count as follower proxy for pages
    if followers_count is not None:
        result["followers_count"] = int(followers_count)
    elif fan_count is not None:
        result["followers_count"] = int(fan_count)
    else:
        result["missing_fields"].append("followers_count")

    # Facebook Pages don't have a "following" count in the same sense
    result["missing_fields"].append("following_count")
    result["notes"].append(
        "Facebook Pages do not expose a 'following count' via the Graph API."
    )

    # Posts: check if the page has any posts
    if posts_data and posts_data.get("data"):
        result["posts_count"] = len(posts_data["data"])  # rough — limited to 1 post query
        result["notes"].append("posts_count is approximate (API limit applied).")
    else:
        result["missing_fields"].append("posts_count")

    # has_bio
    result["has_bio"] = 1 if (bio_text and bio_text.strip()) else 0

    # has_external_link
    result["has_external_link"] = 1 if (website and website.strip()) else 0

    # has_profile_pic — picture data is returned for pages
    if picture_data and not picture_data.get("data", {}).get("is_silhouette", True):
        result["has_profile_pic"] = 1
    else:
        result["has_profile_pic"] = None
        result["missing_fields"].append("has_profile_pic")

    # account_age_bracket — not available via Graph API for pages
    result["missing_fields"].append("account_age_bracket")

    # Facebook Pages are always public
    result["is_private"] = 0
    result["notes"].append(
        "Facebook Pages are always public. Personal profile privacy cannot be "
        "determined via the Graph API."
    )

    return result


def _http_get_json(url: str) -> dict:
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
        "success":           True,
        "username":          username,
        "profile_url":       profile_url,
        "platform":          "facebook",
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

