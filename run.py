"""Start Deer AI:  python run.py  [--host 0.0.0.0] [--port 8000]"""
import argparse
import multiprocessing as mp
import os
import sys

def load_env(path):
    """Read KEY=VALUE lines from deer.env (keeps secrets out of your shell history)."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


PROTECTED = {"deer_data", "deer.env", "venv", ".venv", ".git", "__pycache__", "examples", "_site", ".pytest_cache"}


def _roll_back(app_dir, upd, info, reason):
    """Put the pre-update backup back and start again. Standard library only, so this works even when
    an update broke everything else."""
    import json, time, zipfile
    pending = os.path.join(upd, "pending.json")
    bk = info.get("backup")
    if not bk or not os.path.exists(bk):
        os.remove(pending)
        return
    with zipfile.ZipFile(bk) as z:
        keep = {n for n in z.namelist() if n != ".deer-backup.json"}
        for dirpath, dirnames, files in os.walk(app_dir):
            rel = os.path.relpath(dirpath, app_dir)
            if rel.split(os.sep)[0] in PROTECTED:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in PROTECTED]
            for f in files:
                r = os.path.normpath(os.path.join(rel, f)).replace(os.sep, "/")
                if (rel == "." and f in PROTECTED) or r in keep:
                    continue
                os.remove(os.path.join(dirpath, f))
        for n in keep:
            z.extract(n, app_dir)
    with open(os.path.join(upd, "rolled_back.json"), "w") as f:
        json.dump({"at": time.time(), "reason": reason, "backup": os.path.basename(bk)}, f)
    os.remove(pending)
    print(f"Update rolled back: {reason}", flush=True)
    os.execv(sys.executable, [sys.executable] + sys.argv)


def safety_net(app_dir, data_dir, error=None):
    """After an update, count start attempts; if the new version keeps failing, restore the backup."""
    import json
    upd = os.path.join(data_dir, "updates")
    pending = os.path.join(upd, "pending.json")
    if not os.path.exists(pending):
        return
    try:
        with open(pending) as f:
            info = json.load(f)
    except (OSError, ValueError):
        return
    if error is not None:
        _roll_back(app_dir, upd, info, f"The new version failed to start: {error}")
        return
    info["attempts"] = info.get("attempts", 0) + 1
    if info["attempts"] > 3:
        _roll_back(app_dir, upd, info, "The new version kept stopping right after it started.")
        return
    with open(pending, "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    mp.freeze_support()
    load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deer.env"))
    ap = argparse.ArgumentParser(description="Deer AI server")
    ap.add_argument("--host", default=os.environ.get("DEER_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("DEER_PORT", "8000")))
    ap.add_argument("--data", default=os.environ.get("DEER_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "deer_data")),
                    help="folder for the database, datasets and model weights")
    a = ap.parse_args()
    os.environ["DEER_DATA"] = os.path.abspath(a.data)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    here = os.path.dirname(os.path.abspath(__file__))
    safety_net(here, os.environ["DEER_DATA"])
    import uvicorn
    try:
        from deer.server.app import app
    except Exception as e:  # a broken update: roll back if there's a backup to go back to
        safety_net(here, os.environ["DEER_DATA"], error=f"{type(e).__name__}: {e}")
        raise
    print(f"Deer AI running at http://{a.host}:{a.port}  (data: {os.environ['DEER_DATA']})")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
