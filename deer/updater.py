"""
Update Deer AI from the admin panel, so a phone is enough:

  - upload a new deer-ai.zip, or
  - connect a GitHub repo once (the server makes its own deploy key), then
    Push to GitHub / Update from GitHub, or let a GitHub webhook update it on every push.

Every update backs up the current code first. After an update the server restarts itself (systemd
brings it straight back). If the new version can't start, run.py puts the backup back by itself.
Your data (deer_data), settings (deer.env) and the Python environment (venv) are never touched.
"""
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile

from .server import db

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTECTED = {"deer_data", "deer.env", "venv", ".venv", ".git", "__pycache__", "examples", "_site", ".pytest_cache"}
MAX_ZIP = 60 * 1024 * 1024
KEEP_BACKUPS = 3
# GitHub's published SSH host key (docs.github.com, "GitHub's SSH key fingerprints"), pinned so the
# server only ever talks to the real GitHub.
GITHUB_HOST_KEY = "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl"
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$")

_lock = threading.Lock()


class UpdateError(Exception):
    pass


def upd_dir():
    d = os.path.join(db.DATA_DIR, "updates")
    os.makedirs(d, exist_ok=True)
    return d


def gh_dir():
    d = os.path.join(db.DATA_DIR, "github")
    os.makedirs(d, exist_ok=True)
    return d


def under_systemd():
    return bool(os.environ.get("INVOCATION_ID"))  # systemd sets this for the services it runs


# ================================================================== backups
def _walk_code(root):
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        top = rel.split(os.sep)[0]
        if top in PROTECTED:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in PROTECTED]
        for f in filenames:
            if rel == "." and f in PROTECTED:
                continue
            if f.endswith(".pyc"):
                continue
            yield os.path.normpath(os.path.join(rel, f))


def backup(note=""):
    path = os.path.join(upd_dir(), f"backup-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in _walk_code(APP_DIR):
            z.write(os.path.join(APP_DIR, rel), rel)
        z.writestr(".deer-backup.json", json.dumps({"at": time.time(), "note": note, "version": version()}))
    old = sorted(f for f in os.listdir(upd_dir()) if f.startswith("backup-"))
    for f in old[:-KEEP_BACKUPS]:
        os.remove(os.path.join(upd_dir(), f))
    return path


def backups():
    out = []
    for f in sorted((f for f in os.listdir(upd_dir()) if f.startswith("backup-")), reverse=True):
        p = os.path.join(upd_dir(), f)
        try:
            with zipfile.ZipFile(p) as z:
                info = json.loads(z.read(".deer-backup.json"))
        except (KeyError, ValueError, zipfile.BadZipFile):
            info = {}
        out.append({"file": f, "at": info.get("at", os.path.getmtime(p)), "note": info.get("note", ""),
                    "version": info.get("version", {}), "size_kb": os.path.getsize(p) // 1024})
    return out


def _copy_tree(src_root, files):
    """Write the new code over the old, leaving protected folders (data, settings, venv, .git) alone.
    Code files that no longer exist in the new version are removed."""
    new = set(files)
    for rel in list(_walk_code(APP_DIR)):
        if rel not in new:
            try:
                os.remove(os.path.join(APP_DIR, rel))
            except OSError:
                pass
    for rel in files:
        dst = os.path.join(APP_DIR, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(src_root, rel), dst)


def restore(name=None):
    """Put a backup back (the newest one by default)."""
    items = backups()
    if not items:
        raise UpdateError("There's no backup to go back to yet.")
    pick = next((b for b in items if b["file"] == name), None) if name else items[0]
    if not pick:
        raise UpdateError("That backup doesn't exist.")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(os.path.join(upd_dir(), pick["file"])) as z:
            z.extractall(tmp)
        os.remove(os.path.join(tmp, ".deer-backup.json"))
        files = [os.path.relpath(os.path.join(dp, f), tmp) for dp, _, fs in os.walk(tmp) for f in fs]
        with _lock:
            _copy_tree(tmp, files)
    _save_version({"source": "backup", "restored": pick["file"], "at": time.time()})
    return pick


# ====================================================================== zip
def _safe_members(z):
    for info in z.infolist():
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or ".." in name.split("/") or ":" in name.split("/")[0]:
            raise UpdateError("The zip contains unsafe file paths.")
        yield info, name


def apply_zip(data):
    if len(data) > MAX_ZIP:
        raise UpdateError("That zip is too big for a Deer AI update.")
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise UpdateError("That file isn't a valid zip.")
    members = list(_safe_members(z))
    names = {n for _, n in members}
    roots = [n[: -len("run.py")] for n in names if n.endswith("run.py") and f"{n[:-len('run.py')]}deer/server/app.py" in names]
    if not roots:
        raise UpdateError("This doesn't look like a Deer AI zip (run.py and deer/server/app.py are missing).")
    prefix = min(roots, key=len)
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for info, name in members:
            if info.is_dir() or not name.startswith(prefix):
                continue
            rel = name[len(prefix):]
            if not rel or rel.split("/")[0] in PROTECTED or rel.endswith(".pyc"):
                continue
            dst = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(info) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
            files.append(os.path.normpath(rel))
        return _install(tmp, files, "zip upload")


def _install(src, files, source):
    req_old = _read(os.path.join(APP_DIR, "requirements.txt"))
    req_new = _read(os.path.join(src, "requirements.txt"))
    with _lock:
        bk = backup(f"before {source}")
        _copy_tree(src, files)
        _mark_pending(bk)
    pip = None
    if req_old != req_new:
        pip = install_requirements()
    _save_version({"source": source, "at": time.time(), "files": len(files)})
    return {"files": len(files), "backup": os.path.basename(bk), "requirements_changed": req_old != req_new, "pip": pip}


def _read(p):
    try:
        with open(p) as f:
            return f.read()
    except OSError:
        return ""


def install_requirements():
    import sys
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(APP_DIR, "requirements.txt")],
                       capture_output=True, text=True, timeout=900)
    if r.returncode:
        raise UpdateError("The new version's packages couldn't be installed:\n" + (r.stderr or r.stdout)[-800:])
    return "installed"


# ================================================= self-healing restart marker
def _pending_path():
    return os.path.join(upd_dir(), "pending.json")


def _mark_pending(backup_path):
    with open(_pending_path(), "w") as f:
        json.dump({"backup": backup_path, "attempts": 0, "at": time.time()}, f)


def startup_ok():
    """Called once the server has started: the update worked, so the safety net can go."""
    try:
        os.remove(_pending_path())
    except FileNotFoundError:
        pass


def last_rollback():
    p = os.path.join(upd_dir(), "rolled_back.json")
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def version():
    info = _read(os.path.join(upd_dir(), "version.json")) if os.path.isdir(os.path.join(db.DATA_DIR)) else ""
    v = json.loads(info) if info else {}
    commit = _git("rev-parse", "--short", "HEAD", check=False) if os.path.isdir(os.path.join(APP_DIR, ".git")) else None
    if commit and commit.returncode == 0:
        v["commit"] = commit.stdout.strip()
    return v


def _save_version(v):
    with open(os.path.join(upd_dir(), "version.json"), "w") as f:
        json.dump(v, f)


def restart_soon(delay=1.5):
    """Exit after the reply is sent; systemd starts the new version right away."""
    if not under_systemd():
        return False
    threading.Thread(target=lambda: (time.sleep(delay), os._exit(0)), daemon=True).start()
    return True


# =================================================================== GitHub
def gh_config():
    try:
        with open(os.path.join(gh_dir(), "config.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_gh(cfg):
    with open(os.path.join(gh_dir(), "config.json"), "w") as f:
        json.dump(cfg, f)
    os.chmod(os.path.join(gh_dir(), "config.json"), 0o600)


def _key_path():
    return os.path.join(gh_dir(), "deploy_key")


def public_key():
    try:
        with open(_key_path() + ".pub") as f:
            return f.read().strip()
    except OSError:
        return None


def connect(repo, branch="main"):
    repo = (repo or "").strip().removeprefix("https://github.com/").removesuffix(".git").strip("/")
    if not REPO_RE.match(repo):
        raise UpdateError("Enter the repo as owner/name, for example dashlith/deer-ai.")
    branch = (branch or "main").strip()
    if not re.match(r"^[A-Za-z0-9._/-]{1,100}$", branch):
        raise UpdateError("That branch name isn't valid.")
    if not shutil.which("git") or not shutil.which("ssh-keygen"):
        raise UpdateError("git or ssh isn't installed on the server. Run the installer again, or: apt install -y git openssh-client")
    if not public_key():
        r = subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "deer-ai-server", "-f", _key_path()],
                           capture_output=True, text=True)
        if r.returncode:
            raise UpdateError("Couldn't make a key: " + r.stderr[-300:])
    with open(os.path.join(gh_dir(), "known_hosts"), "w") as f:
        f.write(GITHUB_HOST_KEY + "\n")
    cfg = gh_config()
    cfg.update({"repo": repo, "branch": branch})
    cfg.setdefault("webhook_secret", os.urandom(20).hex())
    cfg.setdefault("auto", False)
    _save_gh(cfg)
    return cfg


def remote_url(cfg=None):
    cfg = cfg or gh_config()
    return cfg.get("remote_url") or f"git@github.com:{cfg['repo']}.git"  # remote_url is only overridden in tests


def _git(*args, check=True, timeout=180):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0",
               GIT_SSH_COMMAND=f"ssh -i {_key_path()} -o IdentitiesOnly=yes -o UserKnownHostsFile={os.path.join(gh_dir(), 'known_hosts')} "
                               f"-o StrictHostKeyChecking=yes -o BatchMode=yes")
    r = subprocess.run(["git", "-C", APP_DIR, *args], capture_output=True, text=True, env=env, timeout=timeout)
    if check and r.returncode:
        raise UpdateError(_explain(r.stderr or r.stdout))
    return r


def _explain(err):
    e = err.strip()
    if "Permission denied" in e or "publickey" in e:
        return "GitHub refused the server's key. Add the key shown on this page to the repo's Deploy keys."
    if "not found" in e.lower() or "does not exist" in e.lower():
        return "GitHub can't find that repo, or the key hasn't been added to it yet."
    if "write access" in e.lower() or "403" in e:
        return "The key can read but not write. Tick 'Allow write access' on the deploy key to push."
    return e[-600:]


def _ensure_repo(cfg):
    if not os.path.isdir(os.path.join(APP_DIR, ".git")):
        _git("init", "-q")
        _git("checkout", "-q", "-b", cfg["branch"], check=False)
    _git("config", "user.name", "Deer AI server")
    _git("config", "user.email", "server@deer-ai.local")
    have = _git("remote", check=False).stdout.split()
    _git("remote", "set-url" if "origin" in have else "add", "origin", remote_url(cfg))


def check(cfg=None):
    """Talk to GitHub: can we reach the repo, and is there anything new?"""
    cfg = cfg or gh_config()
    if not cfg.get("repo"):
        raise UpdateError("Connect a GitHub repo first.")
    r = _git("ls-remote", remote_url(cfg), f"refs/heads/{cfg['branch']}")
    remote = r.stdout.split()[0] if r.stdout.strip() else None
    local = None
    if os.path.isdir(os.path.join(APP_DIR, ".git")):
        lr = _git("rev-parse", "HEAD", check=False)
        local = lr.stdout.strip() if lr.returncode == 0 else None
    return {"reachable": True, "remote_commit": remote, "local_commit": local, "empty": remote is None,
            "update_available": bool(remote and remote != local)}


def push(message="Deer AI from the server"):
    """Publish the server's current code to the (empty or matching) GitHub repo."""
    cfg = gh_config()
    st = check(cfg)
    with _lock:
        _ensure_repo(cfg)
        _git("add", "-A")
        if _git("diff", "--cached", "--quiet", check=False).returncode:
            _git("commit", "-q", "-m", message)
        if not st["empty"]:
            _git("fetch", "-q", "origin", cfg["branch"])
            base = _git("merge-base", "--is-ancestor", f"origin/{cfg['branch']}", "HEAD", check=False)
            if base.returncode:
                raise UpdateError("GitHub has changes the server doesn't. Use Update from GitHub first, then push.")
        _git("push", "-q", "-u", "origin", f"HEAD:refs/heads/{cfg['branch']}")
    return check(cfg)


def pull():
    """Bring the server's code up to date with GitHub."""
    cfg = gh_config()
    st = check(cfg)
    if st["empty"]:
        raise UpdateError("The GitHub repo is empty. Press Push to GitHub to put this server's code there first.")
    if not st["update_available"]:
        return {"updated": False, **st}
    req_old = _read(os.path.join(APP_DIR, "requirements.txt"))
    with _lock:
        bk = backup("before update from GitHub")
        _ensure_repo(cfg)
        _git("fetch", "-q", "origin", cfg["branch"])
        _git("reset", "-q", "--hard", f"origin/{cfg['branch']}")  # ignored files (data, settings, venv) are untouched
        _mark_pending(bk)
    pip = install_requirements() if _read(os.path.join(APP_DIR, "requirements.txt")) != req_old else None
    _save_version({"source": "github", "at": time.time(), "commit": st["remote_commit"][:7]})
    return {"updated": True, "backup": os.path.basename(bk), "pip": pip, **check(cfg)}


def set_auto(on):
    cfg = gh_config()
    cfg["auto"] = bool(on)
    _save_gh(cfg)
    return cfg


def verify_webhook(body, signature):
    secret = gh_config().get("webhook_secret")
    if not secret or not signature or not signature.startswith("sha256="):
        return False
    good = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(good, signature)


def public_status():
    cfg = gh_config()
    return {"app_dir": APP_DIR, "systemd": under_systemd(), "git_installed": bool(shutil.which("git")),
            "version": version(), "backups": backups(), "rolled_back": last_rollback(),
            "github": {"repo": cfg.get("repo"), "branch": cfg.get("branch", "main"), "auto": cfg.get("auto", False),
                       "public_key": public_key(), "webhook_secret": cfg.get("webhook_secret"),
                       "is_git_checkout": os.path.isdir(os.path.join(APP_DIR, ".git"))}}
