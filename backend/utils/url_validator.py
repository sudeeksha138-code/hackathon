"""
URL Validator & Platform Identifier
------------------------------------
Validates that a submitted URL belongs to an allowlisted social media
platform and is a profile URL (not a post, reel, video, etc.).

Security:
  - Only HTTPS URLs accepted.
  - Strict allowlist of platform domains (no wildcard matching).
  - Private/loopback IPs rejected (SSRF protection).
  - Usernames extracted via regex, never by fetching the URL.
"""

import re
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Domain allowlist — only these exact hostnames are permitted
# ---------------------------------------------------------------------------
PLATFORM_DOMAINS = {
    "instagram": {"www.instagram.com", "instagram.com"},
    "facebook":  {"www.facebook.com", "facebook.com", "m.facebook.com"},
    "x":         {"twitter.com", "www.twitter.com", "x.com", "www.x.com"},
}

# Reverse lookup: hostname → platform key
_HOST_TO_PLATFORM: dict[str, str] = {}
for _platform, _hosts in PLATFORM_DOMAINS.items():
    for _host in _hosts:
        _HOST_TO_PLATFORM[_host] = _platform

# ---------------------------------------------------------------------------
# Non-profile path patterns (things we must reject)
# ---------------------------------------------------------------------------
_REJECT_PATH_PATTERNS: dict[str, list[re.Pattern]] = {
    "instagram": [
        re.compile(r"^/p/"),          # posts
        re.compile(r"^/reel/"),       # reels
        re.compile(r"^/stories/"),    # stories
        re.compile(r"^/tv/"),         # IGTV
        re.compile(r"^/explore/"),    # explore
        re.compile(r"^/accounts/"),   # auth pages
    ],
    "facebook": [
        re.compile(r"/posts/"),
        re.compile(r"/videos/"),
        re.compile(r"/photos/"),
        re.compile(r"/groups/"),
        re.compile(r"/events/"),
        re.compile(r"/marketplace/"),
        re.compile(r"/watch"),
        re.compile(r"/pages/"),
    ],
    "x": [
        re.compile(r"/status/"),      # individual tweets
        re.compile(r"/i/"),           # internal X paths
        re.compile(r"/search"),
        re.compile(r"/explore"),
        re.compile(r"/home"),
    ],
}

# ---------------------------------------------------------------------------
# Username extraction patterns per platform
# ---------------------------------------------------------------------------
_USERNAME_PATTERNS: dict[str, re.Pattern] = {
    # instagram.com/{username}/  or  instagram.com/{username}
    "instagram": re.compile(r"^/([A-Za-z0-9._]{1,30})/?$"),
    # facebook.com/{username}  or  facebook.com/profile.php?id=...
    "facebook":  re.compile(r"^/([A-Za-z0-9._\-]{1,100})/?$"),
    # x.com/{username}  or  twitter.com/{username}
    "x":         re.compile(r"^/([A-Za-z0-9_]{1,50})/?$"),
}

# Explicit "not a username" paths on Facebook
_FACEBOOK_RESERVED = {
    "profile.php", "home", "login", "reg", "groups",
    "pages", "events", "marketplace", "watch", "gaming",
    "ad", "ads", "help", "policies", "settings",
}

# Explicit "not a username" paths on X/Twitter
_X_RESERVED = {
    "home", "explore", "notifications", "messages", "settings",
    "i", "login", "signup", "about", "privacy", "tos", "search",
}

# ---------------------------------------------------------------------------
# Private / loopback IP ranges (SSRF protection — just for host-is-IP check)
# ---------------------------------------------------------------------------
_PRIVATE_IP_RE = re.compile(
    r"^(10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|127\.|localhost)",
    re.IGNORECASE,
)


def validate_and_extract(url: str) -> dict:
    """
    Validates a social media profile URL.

    Returns a dict with keys:
      - success (bool)
      - platform (str|None): "instagram", "facebook", or "x"
      - username (str|None)
      - normalized_url (str|None): canonical HTTPS profile URL
      - error (str|None): human-readable rejection reason
    """
    if not url or not isinstance(url, str):
        return _fail("No URL provided.")

    url = url.strip()

    # Must be HTTPS
    if not url.startswith("https://"):
        if url.startswith("http://"):
            # Upgrade silently and continue
            url = "https://" + url[7:]
        else:
            return _fail(
                "Only HTTPS profile URLs are accepted. "
                "Please paste the full URL including https://."
            )

    try:
        parsed = urlparse(url)
    except Exception:
        return _fail("Could not parse the provided URL.")

    hostname = (parsed.hostname or "").lower().strip()

    # SSRF: reject private IPs
    if _PRIVATE_IP_RE.match(hostname):
        return _fail("Private or loopback addresses are not permitted.")

    # Allowlist check
    platform = _HOST_TO_PLATFORM.get(hostname)
    if platform is None:
        return _fail(
            f"Unsupported domain: '{hostname}'. "
            "Only Instagram, Facebook, and X (Twitter) profile URLs are accepted."
        )

    path = parsed.path or "/"
    if not path.endswith("/"):
        path = path + "/"

    # Reject non-profile paths
    for pattern in _REJECT_PATH_PATTERNS.get(platform, []):
        if pattern.search(parsed.path):
            return _fail(
                f"The URL appears to link to a {_path_type_label(parsed.path, platform)}, "
                "not a profile page. Please paste a direct profile URL."
            )

    # Extract username
    username = _extract_username(platform, parsed)
    if username is None:
        return _fail(
            f"Could not extract a username from the {platform.capitalize()} URL. "
            "Please paste a direct profile URL (e.g. https://www.instagram.com/username/)."
        )

    # Build normalized URL
    base_domain = {
        "instagram": "https://www.instagram.com",
        "facebook":  "https://www.facebook.com",
        "x":         "https://x.com",
    }[platform]
    normalized = f"{base_domain}/{username}/"

    return {
        "success":        True,
        "platform":       platform,
        "username":       username,
        "normalized_url": normalized,
        "error":          None,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fail(reason: str) -> dict:
    return {
        "success":        False,
        "platform":       None,
        "username":       None,
        "normalized_url": None,
        "error":          reason,
    }


def _extract_username(platform: str, parsed) -> str | None:
    path = parsed.path.rstrip("/")

    if platform == "facebook":
        # Handle /profile.php?id=XXXXXX — treat as a valid profile reference
        if path == "/profile.php" or path == "":
            uid = parsed.query  # rough — could contain id=
            if "id=" in uid:
                uid_match = re.search(r"id=(\d+)", uid)
                if uid_match:
                    return "id_" + uid_match.group(1)
        # Remove leading slash for matching
        path_stripped = path.lstrip("/")
        if path_stripped.lower() in _FACEBOOK_RESERVED:
            return None
        m = re.match(r"^([A-Za-z0-9._\-]{1,100})$", path_stripped)
        return m.group(1) if m else None

    if platform == "x":
        path_stripped = path.lstrip("/")
        if path_stripped.lower() in _X_RESERVED:
            return None
        m = re.match(r"^([A-Za-z0-9_]{1,50})$", path_stripped)
        return m.group(1) if m else None

    if platform == "instagram":
        path_stripped = path.lstrip("/")
        if not path_stripped:
            return None
        m = re.match(r"^([A-Za-z0-9._]{1,30})$", path_stripped)
        return m.group(1) if m else None

    return None


def _path_type_label(path: str, platform: str) -> str:
    """Human-friendly label for rejected path type."""
    checks = {
        "/p/": "post",
        "/reel/": "reel",
        "/stories/": "story",
        "/status/": "tweet",
        "/posts/": "post",
        "/videos/": "video",
        "/photos/": "photo",
        "/groups/": "group",
        "/events/": "event",
    }
    for fragment, label in checks.items():
        if fragment in path:
            return label
    return "non-profile page"

