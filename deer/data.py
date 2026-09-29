"""Dataset storage, loading and preprocessing for Deer AI."""
import io
import json
import os
import re
import zipfile
import numpy as np
from PIL import Image, ImageOps

from . import chatfmt

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MAX_ZIP_BYTES = int(os.environ.get("DEER_MAX_ZIP_MB", "2048")) * 1024 * 1024
MAX_IMAGES = 200_000
_CLASS_RE = re.compile(r"[^a-zA-Z0-9 _\-]+")


class DatasetError(ValueError):
    pass


# ------------------------------------------------------------------- text
CODE_LANGS = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript", ".ts": "typescript", ".jsx": "jsx", ".tsx": "tsx",
    ".html": "html", ".htm": "html", ".css": "css", ".cs": "csharp", ".lua": "lua", ".gd": "gdscript",
    ".java": "java", ".kt": "kotlin", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".h": "cpp", ".c": "c",
    ".rs": "rust", ".go": "go", ".sh": "bash", ".ps1": "powershell", ".json": "json", ".yml": "yaml",
    ".yaml": "yaml", ".toml": "toml", ".xml": "xml", ".sql": "sql", ".glsl": "glsl", ".hlsl": "hlsl",
    ".shader": "hlsl", ".gdshader": "glsl", ".mcfunction": "mcfunction", ".php": "php", ".rb": "ruby",
    ".swift": "swift", ".dart": "dart", ".vue": "vue", ".svelte": "svelte",
}
PLAIN_EXT = {".txt", ".md", ".csv", ".rst"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", "Library", "Temp", "obj", "bin", "build", "dist", ".godot",
             ".import", ".vs", ".idea", "venv", ".venv", "Logs", "UserSettings"}
MAX_CODE_FILE = 300_000


def _decode(data):
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def code_block(path, text):
    """A source file as training text: its path, then the code in a fenced block."""
    ext = os.path.splitext(path)[1].lower()
    if ext in PLAIN_EXT or ext not in CODE_LANGS:
        return text
    return f"File: {path}\n```{CODE_LANGS[ext]}\n{text.rstrip()}\n```"


def code_zip_to_text(zip_bytes):
    """A zipped project (game, mod, website...) becomes training text, one file after another."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        raise DatasetError("That .zip file couldn't be opened.")
    parts, total = [], 0
    for info in sorted(zf.infolist(), key=lambda i: i.filename):
        name = info.filename.replace("\\", "/")
        segs = [p for p in name.split("/") if p]
        ext = os.path.splitext(name)[1].lower()
        if info.is_dir() or any(p in SKIP_DIRS or p.startswith(".") for p in segs[:-1]):
            continue
        if ext not in CODE_LANGS and ext not in PLAIN_EXT or info.file_size > MAX_CODE_FILE:
            continue
        text = _decode(zf.read(info))
        if text is None or not text.strip():
            continue
        parts.append(code_block("/".join(segs), text))
        total += len(text)
        if total > 50_000_000:
            break
    if not parts:
        raise DatasetError("No code or text files found in the zip.")
    return "\n\n".join(parts)


def save_text_dataset(folder, blobs):
    """blobs: list of (filename, bytes): text, code files, or a .zip of a project. Saved as corpus.txt."""
    os.makedirs(folder, exist_ok=True)
    parts = []
    for name, data in blobs:
        if (name or "").lower().endswith(".zip"):
            parts.append(code_zip_to_text(data))
            continue
        text = _decode(data)
        if text is None:
            text = data.decode("latin-1")
        parts.append(code_block(name or "file.txt", text) if not text.lstrip().startswith("User:") else text)
    text = "\n".join(p.strip("\ufeff") for p in parts)
    if len(text.encode("utf-8")) < 256:
        raise DatasetError("Text dataset needs at least 256 bytes of text.")
    units = chatfmt.parse(text)
    with open(os.path.join(folder, "corpus.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return {"bytes": len(text.encode("utf-8")), "lines": text.count("\n") + 1,
            "chat_format": ("User:" in text and "Deer:" in text),
            "pairs": sum(1 for u in units if u.get("a")), "code_blocks": text.count("```") // 2}


def _training_bytes(path):
    """Read a text file and convert it to training form (every reply ends with the EOT byte)."""
    if not os.path.exists(path):
        return np.zeros(0, dtype=np.int64)
    with open(path, encoding="utf-8", errors="replace") as f:
        text = chatfmt.to_training(f.read())
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)


def load_text(folder):
    return _training_bytes(os.path.join(folder, "corpus.txt"))


def load_learned(folder):
    return _training_bytes(os.path.join(folder, "learned.txt"))


def load_text_split(folder, block):
    """Train on the base corpus plus what users wrote; validate on held-out base text only,
    so the validation score stays comparable as user data accumulates. Approved lessons are
    loaded separately (load_lessons) and mixed into every batch so they're actually learned."""
    base = load_text(folder)
    split = int(len(base) * 0.9)
    train, val = base[:split], base[split:]
    if len(val) <= block + 1:
        train = val = base
    extra = _training_bytes(os.path.join(folder, "user_text.txt"))
    if len(extra):
        train = np.concatenate([train, extra])
    return train, val


def load_lessons(folder, block):
    """Approved lessons, tiled so training windows fit, plus the positions where each lesson starts."""
    lessons = load_learned(folder)
    if len(lessons) < 8:
        return None, None
    reps = max(2, -(-(2 * block + 2) // len(lessons)))
    tok = np.concatenate([lessons] * reps)
    starts = [0] + [i + 1 for i in range(len(tok) - 1) if tok[i] == 10 and tok[i + 1] == ord("U")]
    starts = np.array([s_ for s_ in starts if s_ <= len(tok) - block - 1] or [0])
    return tok, starts


def add_user_text(folder, text):
    """What users write (their own words) becomes extra training text for the chat model."""
    text = text.strip()[:4000]
    if len(text) < 3:
        return
    with open(os.path.join(folder, "user_text.txt"), "a", encoding="utf-8") as f:
        f.write(f"User: {text}\n")


def add_lesson(folder, prompt, answer):
    """A taught answer; line breaks are kept so code lessons stay intact."""
    with open(os.path.join(folder, "learned.txt"), "a", encoding="utf-8") as f:
        f.write(chatfmt.lesson(prompt[:4000], answer[:8000]))


# ----------------------------------------------------------------- images
def _clean_class(name):
    name = _CLASS_RE.sub("", name).strip()
    return name[:60] or None


def save_image_dataset(folder, zip_bytes):
    """Zip layout: <class_name>/<image files>. One wrapping top folder is allowed."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        raise DatasetError("Upload is not a valid .zip file.")
    infos = [i for i in zf.infolist() if not i.is_dir()
             and "__MACOSX" not in i.filename
             and not os.path.basename(i.filename).startswith(".")
             and os.path.splitext(i.filename)[1].lower() in IMAGE_EXT]
    if not infos:
        raise DatasetError("No images found. Put images in folders named after each class.")
    if len(infos) > MAX_IMAGES:
        raise DatasetError(f"Too many images (limit {MAX_IMAGES}).")
    if sum(i.file_size for i in infos) > MAX_ZIP_BYTES:
        raise DatasetError("Unzipped images are too large.")
    paths = [[p for p in i.filename.replace("\\", "/").split("/") if p not in ("", ".", "..")] for i in infos]
    # strip a single shared wrapping folder
    while all(len(p) >= 3 for p in paths) and len({p[0] for p in paths}) == 1:
        paths = [p[1:] for p in paths]
    img_root = os.path.join(folder, "images")
    os.makedirs(img_root, exist_ok=True)
    counts, skipped = {}, 0
    for info, parts in zip(infos, paths):
        cls = _clean_class(parts[0]) if len(parts) >= 2 else "unlabeled"
        if not cls:
            skipped += 1
            continue
        try:
            im = Image.open(io.BytesIO(zf.read(info)))
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((256, 256))
        except Exception:
            skipped += 1
            continue
        d = os.path.join(img_root, cls)
        os.makedirs(d, exist_ok=True)
        n = counts.get(cls, 0)
        im.save(os.path.join(d, f"{n:06d}.png"))
        counts[cls] = n + 1
    if not counts:
        raise DatasetError("None of the images could be read.")
    return {"classes": sorted(counts), "counts": counts, "images": sum(counts.values()), "skipped": skipped}


def add_image(folder, label, pil):
    """Adds one labeled image to an image dataset (new labels create a new class)."""
    cls = _clean_class(label)
    if not cls:
        raise DatasetError("That label has no usable characters.")
    d = os.path.join(folder, "images", cls)
    os.makedirs(d, exist_ok=True)
    n = len(os.listdir(d))
    while os.path.exists(os.path.join(d, f"u{n:06d}.png")):
        n += 1
    im = ImageOps.exif_transpose(pil).convert("RGB")
    im.thumbnail((256, 256))
    im.save(os.path.join(d, f"u{n:06d}.png"))
    for fn in os.listdir(folder):  # resized caches are now stale
        if fn.startswith("cache_") and fn.endswith(".npz"):
            os.remove(os.path.join(folder, fn))
    return cls


def image_stats_for(folder):
    root = os.path.join(folder, "images")
    counts = {c: len(os.listdir(os.path.join(root, c))) for c in sorted(os.listdir(root))
              if os.path.isdir(os.path.join(root, c))}
    return {"classes": sorted(counts), "counts": counts, "images": sum(counts.values()), "skipped": 0}


def square_resize(im, size):
    im = ImageOps.exif_transpose(im).convert("RGB")
    return ImageOps.fit(im, (size, size), Image.BILINEAR)


def load_images(folder, size, with_user=False):
    """Returns X uint8 (N,S,S,3), y int64 (N,), classes list (and, with with_user, a mask of
    pictures users taught). Cached per size."""
    cache = os.path.join(folder, f"cache_{size}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if "user" in z.files:
            out = (z["X"], z["y"], json.loads(str(z["classes"])))
            return out + (z["user"],) if with_user else out
    root = os.path.join(folder, "images")
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    X, y, user = [], [], []
    for ci, c in enumerate(classes):
        for fn in sorted(os.listdir(os.path.join(root, c))):
            with Image.open(os.path.join(root, c, fn)) as im:
                X.append(np.asarray(square_resize(im, size), dtype=np.uint8))
            y.append(ci)
            user.append(fn.startswith("u"))
    X, y, user = np.stack(X), np.array(y, dtype=np.int64), np.array(user, dtype=bool)
    np.savez(cache, X=X, y=y, classes=np.array(json.dumps(classes)), user=user)
    return (X, y, classes, user) if with_user else (X, y, classes)


def to_float(X_uint8):
    """uint8 (N,H,W,3) -> float32 in [0,1], channels-last as the engine expects."""
    return X_uint8.astype(np.float32) / 255.0


# ------------------------------------------------ algorithmic image analysis
def image_stats(im, k=5, seed=0):
    """Non-learned analysis: size, brightness, colorfulness, detail, palette (k-means)."""
    w, h = im.size
    small = np.asarray(ImageOps.exif_transpose(im).convert("RGB").resize((64, 64)), dtype=np.float32)
    px = small.reshape(-1, 3)
    lum = 0.299 * px[:, 0] + 0.587 * px[:, 1] + 0.114 * px[:, 2]
    rg = px[:, 0] - px[:, 1]
    yb = 0.5 * (px[:, 0] + px[:, 1]) - px[:, 2]
    colorful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    g = lum.reshape(64, 64)
    edges = np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()
    # k-means palette
    r = np.random.default_rng(seed)
    cent = px[r.choice(len(px), k, replace=False)]
    for _ in range(12):
        d = ((px[:, None, :] - cent[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                cent[j] = px[lab == j].mean(0)
    share = np.bincount(lab, minlength=k) / len(lab)
    order = np.argsort(-share)
    palette = [{"hex": "#%02x%02x%02x" % tuple(int(c) for c in cent[i]), "share": round(float(share[i]), 3)}
               for i in order]
    return {
        "width": w, "height": h,
        "brightness": round(float(lum.mean() / 255), 3),
        "contrast": round(float(lum.std() / 128), 3),
        "colorfulness": round(colorful / 100, 3),
        "detail": round(float(edges / 32), 3),
        "palette": palette,
    }
