"""Updating from a phone: zip uploads, backups, GitHub push/pull, webhooks, self-healing restarts, installer."""
import hashlib, hmac, io, json, os, shutil, socket, subprocess, sys, tempfile, time, urllib.request, zipfile
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from deer.server import db
from deer import updater as U

IGNORE = shutil.ignore_patterns("deer_data", ".git", "__pycache__", "examples", "_site", ".pytest_cache", "*.pyc")


@pytest.fixture
def app_copy(tmp_path, monkeypatch):
    app = tmp_path / "app"
    shutil.copytree(ROOT, app, ignore=IGNORE)
    (app / "deer_data").mkdir()
    (app / "deer_data" / "precious.db").write_text("my data")
    (app / "deer.env").write_text("SECRET=1")
    (app / "venv" / "lib").mkdir(parents=True)
    (app / "venv" / "lib" / "big.so").write_text("python environment")
    monkeypatch.setattr(U, "APP_DIR", str(app))
    monkeypatch.setattr(db, "DATA_DIR", str(tmp_path / "data"))
    return app


def zip_of(folder, change=None, add=None, drop=None, prefix="deer-ai/"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for dp, dn, fs in os.walk(folder):
            dn[:] = [d for d in dn if d not in ("deer_data", "__pycache__")]
            for f in fs:
                rel = os.path.relpath(os.path.join(dp, f), folder).replace(os.sep, "/")
                if rel in ("deer.env",) or rel == drop:
                    continue
                data = open(os.path.join(dp, f), "rb").read()
                if change and rel == change[0]:
                    data = change[1].encode()
                z.writestr(prefix + rel, data)
        if add:
            z.writestr(prefix + add[0], add[1])
        z.writestr(prefix + "deer_data/precious.db", "HACKED")  # data inside a zip is never used
    return buf.getvalue()


def test_zip_update_backup_and_restore(app_copy):
    new = zip_of(app_copy, change=("README.md", "new readme"), add=("tools/new_tool.py", "print(1)"), drop="deer.env.example")
    res = U.apply_zip(new)
    assert (app_copy / "README.md").read_text() == "new readme" and (app_copy / "tools/new_tool.py").exists()
    assert not (app_copy / "deer.env.example").exists()
    assert (app_copy / "deer_data/precious.db").read_text() == "my data" and (app_copy / "deer.env").read_text() == "SECRET=1"
    assert res["backup"].startswith("backup-") and not res["requirements_changed"]
    assert os.path.exists(os.path.join(db.DATA_DIR, "updates", "pending.json"))
    U.startup_ok()
    assert not os.path.exists(os.path.join(db.DATA_DIR, "updates", "pending.json"))
    U.restore()  # back to how it was
    assert (app_copy / "README.md").read_text() != "new readme" and not (app_copy / "tools/new_tool.py").exists()
    assert (app_copy / "deer.env.example").exists() and (app_copy / "deer_data/precious.db").read_text() == "my data"
    for i in range(4):
        U.backup(f"b{i}")
    assert len(U.backups()) == 3  # only the last few are kept


def test_bad_zips(app_copy):
    with pytest.raises(U.UpdateError, match="valid zip"):
        U.apply_zip(b"nope")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", "x")
    with pytest.raises(U.UpdateError, match="Deer AI zip"):
        U.apply_zip(buf.getvalue())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("deer-ai/run.py", "x"); z.writestr("deer-ai/deer/server/app.py", "x"); z.writestr("../../etc/evil", "x")
    with pytest.raises(U.UpdateError, match="unsafe"):
        U.apply_zip(buf.getvalue())


def test_github_push_pull(app_copy, tmp_path):
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)
    U.connect("https://github.com/dashlith/deer-ai.git", "main")
    cfg = U.gh_config()
    assert cfg["repo"] == "dashlith/deer-ai" and U.public_key().startswith("ssh-ed25519 ")
    cfg["remote_url"] = str(bare)  # stand-in for github.com
    U._save_gh(cfg)
    with pytest.raises(U.UpdateError, match="empty"):
        U.pull()
    assert U.check()["empty"]
    U.push()
    st = U.check()
    assert not st["empty"] and not st["update_available"]
    files = subprocess.run(["git", "--git-dir", str(bare), "ls-tree", "-r", "--name-only", "main"], capture_output=True, text=True).stdout
    assert "run.py" in files and "deer_data" not in files and "deer.env\n" not in files + "\n".replace("deer.env.example", "")
    assert "precious" not in files and "\ndeer.env\n" not in "\n" + files and "venv/" not in files

    # someone changes the code on GitHub
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(bare), str(work)], check=True)
    (work / "README.md").write_text("changed on github")
    subprocess.run(["git", "-C", str(work), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "edit"], check=True)
    subprocess.run(["git", "-C", str(work), "push", "-q"], check=True)
    assert U.check()["update_available"]
    res = U.pull()
    assert res["updated"] and (app_copy / "README.md").read_text() == "changed on github"
    assert (app_copy / "deer_data/precious.db").read_text() == "my data" and (app_copy / "deer.env").read_text() == "SECRET=1"
    assert U.pull()["updated"] is False
    # GitHub ahead of a server edit: pushing is refused instead of overwriting GitHub
    (work / "README.md").write_text("again"); subprocess.run(["git", "-C", str(work), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "e2"], check=True)
    subprocess.run(["git", "-C", str(work), "push", "-q"], check=True)
    (app_copy / "notes.txt").write_text("server edit")
    with pytest.raises(U.UpdateError, match="Update from GitHub first"):
        U.push()
    for bad in ("not a repo", "a/b/c", "x"):
        with pytest.raises(U.UpdateError):
            U.connect(bad)


def test_webhook_signature(app_copy):
    U.connect("dashlith/deer-ai")
    secret = U.gh_config()["webhook_secret"]
    body = b'{"ref": "refs/heads/main"}'
    good = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert U.verify_webhook(body, good)
    assert not U.verify_webhook(body + b" ", good) and not U.verify_webhook(body, "sha256=00") and not U.verify_webhook(body, None)


def _port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def test_broken_update_rolls_back_by_itself(app_copy, tmp_path):
    data = tmp_path / "data"
    bk = U.backup("good version")
    U._mark_pending(bk)
    app_py = app_copy / "deer" / "server" / "app.py"
    good = app_py.read_text()
    app_py.write_text("raise RuntimeError('broken update')\n" + good)
    port = _port()
    proc = subprocess.Popen([sys.executable, "run.py", "--port", str(port), "--data", str(data)], cwd=app_copy,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        ok = False
        for _ in range(80):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/auth/status", timeout=1); ok = True; break
            except OSError:
                time.sleep(0.5)
        assert ok, "server didn't come back after rolling back"
        assert app_py.read_text() == good
        rb = json.load(open(data / "updates" / "rolled_back.json"))
        assert "broken update" in rb["reason"] and not (data / "updates" / "pending.json").exists()
    finally:
        proc.terminate(); proc.wait(timeout=20)


def test_installer_dry_run():
    r = subprocess.run(["bash", "tools/install_vps.sh", "ai.example.com"], cwd=ROOT, env=dict(os.environ, DRY_RUN="1"),
                       capture_output=True, text=True)
    assert r.returncode == 0 and "deer-ai.service" in r.stdout and "ai.example.com" in r.stdout
    r = subprocess.run(["bash", "tools/install_vps.sh", "bad domain;rm -rf"], cwd=ROOT, env=dict(os.environ, DRY_RUN="1"),
                       capture_output=True, text=True)
    assert r.returncode == 1 and "doesn't look like a domain" in r.stdout


def test_update_endpoints(tmp_path):
    from fastapi.testclient import TestClient
    from deer.server import app as APP
    db.DATA_DIR = str(tmp_path / "srv")
    H = lambda t: {"Authorization": f"Bearer {t}"}
    with TestClient(APP.app) as c:
        A = c.post("/api/auth/register", json={"username": "admin", "password": "adminadmin"}).json()["token"]
        Uu = c.post("/api/auth/register", json={"username": "user1", "password": "useruser1"}).json()["token"]
        assert c.get("/api/admin/update", headers=H(Uu)).status_code == 403
        st = c.get("/api/admin/update", headers=H(A)).json()
        assert st["github"]["repo"] is None and st["systemd"] is False and st["training"] is False
        r = c.post("/api/admin/update/zip", headers=H(A), files={"file": ("x.zip", b"nope", "application/zip")})
        assert r.status_code == 400 and "valid zip" in r.json()["detail"]
        assert c.post("/api/admin/update/rollback", headers=H(A), json={}).status_code == 400  # no backups yet
        st = c.put("/api/admin/update/github", headers=H(A), json={"repo": "dashlith/deer-ai"}).json()
        assert st["github"]["public_key"].startswith("ssh-ed25519") and st["github"]["webhook_secret"]
        secret = st["github"]["webhook_secret"]
        sign = lambda b: "sha256=" + hmac.new(secret.encode(), b, hashlib.sha256).hexdigest()
        body = b'{"zen": "hi"}'
        assert c.post("/api/github/webhook", content=body, headers={"x-hub-signature-256": "sha256=bad", "x-github-event": "ping"}).status_code == 401
        assert c.post("/api/github/webhook", content=body, headers={"x-hub-signature-256": sign(body), "x-github-event": "ping"}).json()["ok"]
        push = b'{"ref": "refs/heads/main"}'
        r = c.post("/api/github/webhook", content=push, headers={"x-hub-signature-256": sign(push), "x-github-event": "push"}).json()
        assert r["ignored"] and "off" in r["reason"]  # automatic updates are off until switched on
        other = b'{"ref": "refs/heads/dev"}'
        c.patch("/api/admin/update/github", headers=H(A), json={"auto": True})
        assert c.post("/api/github/webhook", content=other, headers={"x-hub-signature-256": sign(other), "x-github-event": "push"}).json()["ignored"]
