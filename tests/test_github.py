"""GitHub hosting: the Pages build, relative paths, cross-site access, workflows and Codespaces config."""
import glob, json, os, re, socket, subprocess, sys, tempfile, time, urllib.request
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_pages


def test_pages_build(tmp_path):
    out = tmp_path / "site"
    assert build_pages.build("https://ai.dashlith.com/", str(out)) == "https://ai.dashlith.com"
    for f in ("index.html", "admin.html", "plans.html", "developers.html", "404.html", "sw.js", "manifest.webmanifest",
              ".nojekyll", "static/common.js", "static/style.css", "static/icons/icon-192.png"):
        assert (out / f).exists(), f
    assert 'window.DEER_API = "https://ai.dashlith.com"' in (out / "static/config.js").read_text()
    assert not (out / "static/index.html").exists()
    for bad in ("http://ai.dashlith.com", "ai.dashlith.com", "https://ai.dashlith.com/deer"):
        with pytest.raises(SystemExit):
            build_pages.build(bad, str(tmp_path / "x"))


def test_frontend_paths_work_anywhere():
    """No page or script may assume it's at the site root, and every server call goes through API_BASE."""
    for f in glob.glob(os.path.join(ROOT, "static", "*.html")) + glob.glob(os.path.join(ROOT, "static", "*.js")):
        s = open(f).read()
        assert not re.search(r'(href|src)="/(?!/)', s), f
        assert 'location.href = "/"' not in s, f
    for f in glob.glob(os.path.join(ROOT, "static", "*.js")):
        for m in re.finditer(r"fetch\(([^,)]*)", open(f).read()):
            arg = m.group(1)
            assert "API_BASE" in arg or f.endswith("sw.js") or "${base}" in arg, (f, arg)
    man = json.load(open(os.path.join(ROOT, "static", "manifest.webmanifest")))
    assert man["start_url"] == "./" and all(not i["src"].startswith("/") for i in man["icons"])


def test_workflows_and_codespaces():
    dep = yaml.safe_load(open(os.path.join(ROOT, ".github/workflows/deploy.yml")))
    assert set(dep["jobs"]) == {"test", "deploy"} and dep["jobs"]["deploy"]["needs"] == "test"
    run = json.dumps(dep["jobs"]["deploy"])
    for keep in ("deer_data", "deer.env", "venv"):
        assert f"--exclude {keep}" in run  # the server's data, settings and venv are never overwritten
    assert "systemctl restart deer-ai" in run and "api/auth/status" in run
    pages = yaml.safe_load(open(os.path.join(ROOT, ".github/workflows/pages.yml")))
    assert "build_pages.py" in json.dumps(pages["jobs"]["build"]) and pages["jobs"]["deploy"]["needs"] == "build"
    dc = json.load(open(os.path.join(ROOT, ".devcontainer/devcontainer.json")))
    assert 8000 in dc["forwardPorts"] and "run.py" in dc["postAttachCommand"]


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def test_cross_site_access_and_page_routes():
    port = _free_port()
    env = dict(os.environ, DEER_DATA=tempfile.mkdtemp(), DEER_ALLOWED_ORIGINS="https://deer.github.io")
    proc = subprocess.Popen([sys.executable, "run.py", "--port", str(port)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(base + "/api/auth/status", timeout=1); break
            except OSError:
                time.sleep(0.5)

        def preflight(origin):
            req = urllib.request.Request(base + "/api/auth/login", method="OPTIONS", headers={
                "Origin": origin, "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type"})
            try:
                return urllib.request.urlopen(req, timeout=5).headers
            except urllib.error.HTTPError as e:
                return e.headers
        assert preflight("https://deer.github.io").get("access-control-allow-origin") == "https://deer.github.io"
        assert preflight("https://evil.example").get("access-control-allow-origin") is None
        for page in ("admin.html", "plans.html", "developers.html", "index.html"):
            body = urllib.request.urlopen(f"{base}/{page}", timeout=5).read().decode()
            assert "static/config.js" in body
        assert b"DEER_API" in urllib.request.urlopen(base + "/static/config.js", timeout=5).read()
        with pytest.raises(urllib.error.HTTPError):  # no way to ask for other files through the page routes
            urllib.request.urlopen(base + "/..%2Fdeer%2Fserver%2Fapp.html", timeout=5)
    finally:
        proc.terminate(); proc.wait(timeout=20)
