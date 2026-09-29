"""
Builds the website for GitHub Pages: the chat, admin, plans and API pages as static files that talk
to your Deer AI server (the AI itself always runs on your server).

    python tools/build_pages.py --api https://ai.dashlith.com --out _site

The GitHub Actions workflow runs this for you. On the server, set
DEER_ALLOWED_ORIGINS=https://yourname.github.io so the pages are allowed to call it.
"""
import argparse
import os
import shutil
import sys
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "static")
PAGES = ("index.html", "admin.html", "plans.html", "developers.html")


def build(api, out):
    u = urlparse(api or "")
    if u.scheme != "https" or not u.netloc:
        raise SystemExit("--api must be your server's https:// address, for example https://ai.dashlith.com "
                         "(GitHub Pages is https, and browsers block calls from https pages to http servers).")
    if u.path not in ("", "/"):
        raise SystemExit("--api should be just the address, without a path.")
    api = f"{u.scheme}://{u.netloc}"
    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(STATIC, os.path.join(out, "static"), ignore=shutil.ignore_patterns("*.html", "sw.js", "manifest.webmanifest"))
    for name in PAGES + ("sw.js", "manifest.webmanifest"):
        shutil.copy(os.path.join(STATIC, name), os.path.join(out, name))
    with open(os.path.join(out, "static", "config.js"), "w") as f:
        f.write(f'// Built for GitHub Pages: every request goes to your Deer AI server.\nwindow.DEER_API = "{api}";\n')
    shutil.copy(os.path.join(out, "index.html"), os.path.join(out, "404.html"))  # unknown paths show the chat
    open(os.path.join(out, ".nojekyll"), "w").close()  # serve files as-is
    return api


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", required=True, help="your Deer AI server, e.g. https://ai.dashlith.com")
    ap.add_argument("--out", default="_site")
    a = ap.parse_args()
    print(f"Built {a.out}/ for GitHub Pages, talking to {build(a.api, a.out)}")
    sys.exit(0)
