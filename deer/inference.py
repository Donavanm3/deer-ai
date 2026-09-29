"""Loads trained checkpoints for serving; reloads automatically when training writes a new one."""
import base64
import io
import os
import re
import threading
import time
import numpy as np
from PIL import Image

from .engine import Tensor, xp
from .models import build_model
from . import data as D
from . import chatfmt
from .server import db

_cache = {}
_lock = threading.Lock()


class NotReady(Exception):
    pass


def get(mid):
    row = db.model_row(mid)
    if not row:
        raise NotReady("Model not found.")
    wpath = os.path.join(db.model_dir(mid), "weights.npz")
    if not os.path.exists(wpath):
        raise NotReady(f"'{row['name']}' has not been trained yet.")
    mtime = os.path.getmtime(wpath)
    with _lock:
        hit = _cache.get(mid)
        if hit and hit[0] == mtime:
            return hit[1], row
        n_classes = len(row["meta"].get("classes", []))
        model = build_model(row["kind"], row["config"], n_classes)
        with np.load(wpath) as z:
            model.load_state_dict({k: z[k] for k in z.files})
        _cache[mid] = (mtime, model)
        return model, row


def forget(mid):
    with _lock:
        _cache.pop(mid, None)


REPLY_SECONDS = float(os.environ.get("DEER_REPLY_SECONDS", "30"))  # hard cap per reply
MAX_REPLY = int(os.environ.get("DEER_MAX_REPLY", "1500"))          # bytes; room for code
STOPS = [bytes([chatfmt.EOT_BYTE]), b"\nUser:"]  # end-of-reply byte; older one-line models stop at the next turn


def build_prompt(history, text, block_size):
    prompt = ""
    for role, content in history[-8:]:
        prompt += f"User: {content}\n" if role == "user" else f"Deer: {content}{chatfmt.EOT}\n"
    prompt += f"User: {text}\nDeer:"
    return prompt.encode("utf-8")[-block_size:]


def chat_stream(mid, history, text, temperature=0.8, max_new=None):
    """Yields the reply piece by piece, until the model ends its reply (it can span many lines, e.g. code)."""
    model, row = get(mid)
    pb = build_prompt(history, text, row["config"]["block_size"])
    return model.stream(pb, max_new=min(max_new or MAX_REPLY, MAX_REPLY), temperature=temperature, top_k=40,
                        stop=STOPS, deadline=time.monotonic() + REPLY_SECONDS)


def prompt_tokens(mid, history, text):
    """How many tokens the model actually reads for this request (Deer models use one token per byte)."""
    _, row = get(mid)
    return len(build_prompt(history, text, row["config"]["block_size"]))


def chat_reply(mid, history, text, temperature=0.8):
    return "".join(chat_stream(mid, history, text, temperature)).strip() or "…"


def analyze(mid, im, top=5):
    model, row = get(mid)
    classes = row["meta"]["classes"]
    arr = np.asarray(D.square_resize(im, row["config"]["image_size"]), dtype=np.uint8)[None]
    probs = model.predict(Tensor(xp.asarray(D.to_float(arr))))[0]
    order = np.argsort(-probs)[:top]
    return [{"label": classes[i], "confidence": round(float(probs[i]), 4)} for i in order]


def match_class(prompt, classes):
    words = set(re.findall(r"[a-z0-9]+", prompt.lower()))
    best, best_score = None, 0
    for i, c in enumerate(classes):
        toks = set(re.findall(r"[a-z0-9]+", c.lower()))
        score = len(words & toks) + (2 if c.lower() in prompt.lower() else 0)
        if score > best_score:
            best, best_score = i, score
    return best


def imagine(mid, prompt, count=4, temperature=1.0, seed=None):
    model, row = get(mid)
    classes = row["meta"]["classes"]
    ci = match_class(prompt, classes)
    matched = ci is not None
    r = np.random.default_rng(seed)
    ids = [ci if matched else int(r.integers(len(classes))) for _ in range(count)]
    imgs = model.sample(ids, temperature=temperature, seed=None if seed is None else seed + 1)
    out = []
    for i, img in enumerate(imgs):
        pil = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
        pil = pil.resize((256, 256), Image.BICUBIC)
        out.append({"label": classes[ids[i]], "data_url": to_data_url(pil)})
    return out, matched


def to_data_url(pil, fmt="PNG"):
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()
