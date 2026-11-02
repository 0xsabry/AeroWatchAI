"""Phase 5 — license-verified image pool from Wikimedia Commons.

Downloads candidate images for the four AeroWatch classes plus hard
negatives, accepting ONLY files whose Commons metadata carries a
license that permits reuse with attribution:

  allowed: CC0, Public domain, CC BY (any version), CC BY-SA (any)
  rejected: CC BY-NC*, CC BY-ND*, all-rights-reserved, unknown

Every accepted file's provenance (title, page URL, author, license,
license URL, source URL, SHA-256) is recorded in pool_manifest.json —
this is the dataset's license evidence, not a claim made elsewhere.

Usage: python evaluation/fetch_commons_pool.py --out ~/pool_dir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "AeroWatchAI-dataset-research/1.0 (educational project)"}

QUERIES: dict[str, list[str]] = {
    "drone": ["quadcopter in flight", "drone flying sky", "multirotor UAV flight"],
    "aircraft": ["airliner in flight", "jet aircraft flying", "airplane in the sky"],
    "bird": ["bird in flight", "seagull flying", "flock of birds flying"],
    "balloon": ["hot air balloon flight", "hot air balloons festival"],
    "negative": ["clouds sky", "mountain landscape panorama", "city skyline", "ocean waves"],
}


def api_get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def license_allowed(meta: dict) -> tuple[bool, str, str]:
    """Return (allowed, license_short_name, license_url) from extmetadata."""
    lic = (meta.get("LicenseShortName") or {}).get("value", "").strip()
    lic_url = (meta.get("LicenseUrl") or {}).get("value", "")
    low = lic.lower()
    if low.startswith("cc0") or low == "public domain":
        return True, lic, lic_url
    if low.startswith("cc by-sa") or low == "cc by" or low.startswith("cc by "):
        if "nc" in low or "nd" in low:
            return False, lic, lic_url
        return True, lic, lic_url
    return False, lic, lic_url


def search_titles(query: str, limit: int) -> list[str]:
    data = api_get(
        {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": 6,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            "iiurlwidth": 1024,
        }
    )
    pages = (data.get("query") or {}).get("pages") or {}
    out = []
    for page in pages.values():
        ii = (page.get("imageinfo") or [{}])[0]
        meta = ii.get("extmetadata") or {}
        ok, lic, lic_url = license_allowed(meta)
        if not ok:
            continue
        if ii.get("width", 0) < 640 or ii.get("height", 0) < 480:
            continue
        out.append(
            {
                "title": page.get("title"),
                "page_url": ii.get("descriptionurl"),
                "download_url": ii.get("thumburl") or ii.get("url"),
                "artist": ((meta.get("Artist") or {}).get("value") or "")[:200],
                "license": lic,
                "license_url": lic_url,
                "width": ii.get("width"),
                "height": ii.get("height"),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-query", type=int, default=18)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    seen_titles: set[str] = set()
    for klass, queries in QUERIES.items():
        kdir = out_dir / klass
        kdir.mkdir(exist_ok=True)
        n = 0
        for q in queries:
            try:
                results = search_titles(q, args.per_query)
            except Exception as e:
                print(f"[{klass}] query '{q}' failed: {e}")
                continue
            for item in results:
                if item["title"] in seen_titles:
                    continue
                seen_titles.add(item["title"])
                fname = f"{klass}_{n:03d}.jpg"
                dest = kdir / fname
                try:
                    req = urllib.request.Request(item["download_url"], headers=UA)
                    with urllib.request.urlopen(req, timeout=60) as r:
                        blob = r.read()
                    dest.write_bytes(blob)
                except Exception as e:
                    print(f"[{klass}] download failed {item['title']}: {e}")
                    continue
                item.update(
                    {
                        "class_query": klass,
                        "file": str(dest.relative_to(out_dir)),
                        "sha256": hashlib.sha256(blob).hexdigest(),
                        "bytes": len(blob),
                    }
                )
                manifest.append(item)
                n += 1
                time.sleep(0.2)
            print(f"[{klass}] '{q}' -> cumulative {n}")
    (out_dir / "pool_manifest.json").write_text(json.dumps(manifest, indent=1))
    from collections import Counter

    print("POOL:", dict(Counter(m["class_query"] for m in manifest)))
    print("LICENSES:", dict(Counter(m["license"] for m in manifest)))


if __name__ == "__main__":
    main()
