"""
Training worker. Runs in its own process so the web server stays responsive.
Checkpoints are written atomically every eval interval, so training can be
stopped at any time and resumed later from the last checkpoint.
"""
import json
import os
import time
import traceback
import numpy as np
from PIL import Image

from .engine import Tensor, no_grad, xp, cross_entropy, soft_cross_entropy, to_numpy as to_np
from .engine.nn import seed as nn_seed
from .engine.optim import AdamW, cosine_lr
from .models import build_model
from . import data as D
from .server import db


class ThreadShare:
    """Uses the number of CPU threads the scheduler assigns (so two jobs share the cores instead of
    fighting over them). Re-checked while training, since jobs start and finish at different times."""
    def __init__(self, mid):
        self.path = os.path.join(db.model_dir(mid), "THREADS")
        self.current = None
        try:
            from threadpoolctl import threadpool_limits  # small helper that sets NumPy's math threads
            self._limit = threadpool_limits
        except ImportError:
            self._limit = None

    def check(self):
        if self._limit is None:
            return
        try:
            with open(self.path) as f:
                n = max(1, int(f.read().strip()))
        except (OSError, ValueError):
            return
        if n != self.current:
            self._limit(limits=n)
            self.current = n


def stop_flag(mid):
    return os.path.join(db.model_dir(mid), "STOP")


def _save_npz(path, arrays):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def save_checkpoint(mid, model, opt):
    d = db.model_dir(mid)
    os.makedirs(d, exist_ok=True)
    _save_npz(os.path.join(d, "optim.npz"), opt.state_dict())
    _save_npz(os.path.join(d, "weights.npz"), model.state_dict())  # written last: signals a complete checkpoint


def load_weights(mid):
    p = os.path.join(db.model_dir(mid), "weights.npz")
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def _update(mid, **fields):
    fields["updated_at"] = db.now()
    keys = ", ".join(f"{k}=?" for k in fields)
    db.ex(f"UPDATE models SET {keys} WHERE id=?", (*fields.values(), mid))


def grid_png(imgs, path, cols=4, scale=3):
    """imgs: (N,S,S,3) in [0,1] -> tiled PNG."""
    n, s, _, _ = imgs.shape
    rows = (n + cols - 1) // cols
    canvas = np.ones((rows * (s + 2), cols * (s + 2), 3), dtype=np.float32)
    for i in range(n):
        r, c = divmod(i, cols)
        canvas[r * (s + 2) + 1:r * (s + 2) + 1 + s, c * (s + 2) + 1:c * (s + 2) + 1 + s] = imgs[i]
    im = Image.fromarray((canvas * 255).clip(0, 255).astype(np.uint8))
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)


def run(model_id, data_dir, extra_steps, lr_scale=1.0, auto=False):
    """Process entry point. `auto` runs come from the auto-learn scheduler: they fine-tune gently
    (lr_scale) and are rolled back automatically if the model gets worse."""
    db.DATA_DIR = data_dir
    import multiprocessing
    if multiprocessing.current_process().name != "MainProcess":
        try:
            os.nice(10)  # chat and the website always get the CPU first
        except (AttributeError, OSError):
            pass
    try:
        _run(model_id, int(extra_steps), float(lr_scale), bool(auto))
    except Exception as e:  # report failure to the admin panel
        traceback.print_exc()
        _update(model_id, status="error", error=f"{type(e).__name__}: {e}"[:500])


def backup_checkpoint(mid, step, meta):
    d = db.model_dir(mid)
    if os.path.exists(os.path.join(d, "weights.npz")):
        with open(os.path.join(d, "backup.json"), "w") as f:
            json.dump({"step": step, "meta": meta}, f)
    for name in ("weights", "optim"):
        src = os.path.join(d, f"{name}.npz")
        if os.path.exists(src):
            with open(src, "rb") as f, open(os.path.join(d, f"{name}.prev.npz.tmp"), "wb") as g:
                g.write(f.read())
            os.replace(os.path.join(d, f"{name}.prev.npz.tmp"), os.path.join(d, f"{name}.prev.npz"))


def restore_checkpoint(mid):
    """Put the pre-training checkpoint back (weights, step, classes). Returns the restored
    info, or None if there's no backup."""
    d = db.model_dir(mid)
    if not os.path.exists(os.path.join(d, "weights.prev.npz")):
        return None
    info = {"step": None, "meta": None}
    bj = os.path.join(d, "backup.json")
    if os.path.exists(bj):
        with open(bj) as f:
            info = json.load(f)
        os.remove(bj)
    for name in ("optim", "weights"):
        prev = os.path.join(d, f"{name}.prev.npz")
        if os.path.exists(prev):
            os.replace(prev, os.path.join(d, f"{name}.npz"))
    if info["step"] is not None:
        db.ex("DELETE FROM training_log WHERE model_id=? AND step>?", (mid, info["step"]))
    return info


def remap_classes(kind, sd, old, new, cfg):
    """Re-point a model's class outputs from `old` classes to `new` ones. Classes in both keep
    everything learned; new classes start fresh; classes that disappeared are dropped."""
    pairs = [(i, new.index(c)) for i, c in enumerate(old) if c in new]
    oi = [p[0] for p in pairs]
    ni = [p[1] for p in pairs]
    K = len(new)
    r = np.random.default_rng(0)
    sd = dict(sd)
    if kind == "vision":
        w, b = sd["fc.w"], sd["fc.b"]
        nw = r.normal(0, 0.01, (w.shape[0], K)).astype(np.float32)
        nb = np.zeros(K, dtype=np.float32)
        nw[:, ni], nb[ni] = w[:, oi], b[oi]
        sd["fc.w"], sd["fc.b"] = nw, nb
    elif kind == "canvas":
        for key, lead in (("eh.w", sd["eh.w"].shape[0] - len(old)), ("dh.w", int(cfg["latent"]))):
            w = sd[key]
            nw = np.zeros((lead + K, w.shape[1]), dtype=np.float32)
            nw[:lead] = w[:lead]
            nw[[lead + j for j in ni]] = w[[lead + i for i in oi]]
            sd[key] = nw
    return sd


def warm_start(kind, model, src_id, classes, cfg):
    """Start a new model from another trained model's weights. Layers with matching shapes are
    copied (so a deeper or wider model still inherits what fits); the rest start fresh."""
    src = db.model_row(src_id)
    if not src or src["kind"] != kind:
        raise ValueError("The model to start from is missing or a different type.")
    ssd = load_weights(src_id)
    if ssd is None:
        raise ValueError(f"'{src['name']}' hasn't been trained, so there's nothing to start from.")
    if kind != "chat" and src["meta"].get("classes"):
        ssd = remap_classes(kind, ssd, src["meta"]["classes"], classes, src["config"])
    copied, total = 0, 0
    for name, p in model.named_parameters():
        total += 1
        a = ssd.get(name)
        if a is None:
            continue
        if tuple(a.shape) == tuple(p.shape):
            p.data = xp.asarray(a, dtype=np.float32)
            copied += 1
        elif name == "pos.w" and a.shape[1] == p.shape[1]:  # longer/shorter context: copy what overlaps
            n = min(a.shape[0], p.shape[0])
            new = to_np(p.data).copy()
            new[:n] = a[:n]
            p.data = xp.asarray(new)
            copied += 1
    if not copied:
        raise ValueError(f"Nothing could be copied from '{src['name']}': its sizes don't match this model.")
    return {"from": src["name"], "copied": copied, "total": total}


def load_teacher(kind, teacher_id, classes, cfg):
    t = db.model_row(teacher_id)
    if not t or t["kind"] != kind or kind == "canvas":
        raise ValueError("The teacher model is missing or can't teach this type of model.")
    sd = load_weights(teacher_id)
    if sd is None:
        raise ValueError(f"Teacher '{t['name']}' hasn't been trained yet.")
    tcls = t["meta"].get("classes", [])
    teacher = build_model(kind, t["config"], len(tcls))
    teacher.load_state_dict(sd)
    cmap = None
    if kind == "chat" and t["config"]["block_size"] < int(cfg["block_size"]):
        raise ValueError(f"Teacher '{t['name']}' reads less context than this model. Pick a teacher with a block_size of "
                         f"at least {cfg['block_size']}.")
    if kind == "vision":
        cmap = np.array([tcls.index(c) if c in tcls else -1 for c in classes])
        if (cmap < 0).all():
            raise ValueError(f"Teacher '{t['name']}' doesn't know any of this dataset's classes.")
    return teacher, cmap, t["name"]


def teacher_probs(kind, teacher, cmap, xb, yb, temp):
    with no_grad():
        lg = to_np(teacher.forward(xb).data).astype(np.float64)
    if kind == "chat":
        lg = lg.reshape(-1, lg.shape[-1])
    lg = lg / temp
    p = np.exp(lg - lg.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    if kind == "vision":  # re-order to this model's classes; unknown classes fall back to the true label
        full = np.where(cmap[None, :] >= 0, p[:, np.maximum(cmap, 0)], 0.0)
        rows = full.sum(1)
        hard = np.eye(len(cmap))[yb]
        full = np.where(rows[:, None] > 1e-6, full / np.maximum(rows[:, None], 1e-12), hard)
        p = full
    return p.astype(np.float32)


def _worse(kind, before, after):
    if before is None or after is None:
        return False
    if kind == "vision":
        return after < before - 0.03          # accuracy dropped more than 3 points
    if kind == "chat":                        # held-out loss rose by more than 8% AND more than 0.05
        return after - before > max(0.05, 0.08 * before)
    return after > before * 1.05              # image creation: reconstruction loss rose more than 5%


def _run(mid, extra_steps, lr_scale=1.0, auto=False):
    row = db.model_row(mid)
    kind, cfg = row["kind"], row["config"]
    ds_dir = db.dataset_dir(row["dataset_id"])
    mdir = db.model_dir(mid)
    os.makedirs(mdir, exist_ok=True)
    # (a leftover stop signal is cleared by the scheduler before this worker starts, so a Stop
    #  pressed while this worker is still starting up is honoured below)
    nn_seed(1234 + mid)
    r = np.random.default_rng(int(time.time()))
    meta = dict(row["meta"])
    meta_before = dict(meta)
    B = int(cfg["batch_size"])

    # ------------------------------------------------------------ data
    if kind == "chat":
        T = int(cfg["block_size"])
        if len(D.load_text(ds_dir)) < T * 4:
            raise ValueError(f"Text dataset is too small for block_size {T}; add more text or lower block_size.")
        train_tok, val_tok = D.load_text_split(ds_dir, T)
        lesson_tok, lesson_starts = D.load_lessons(ds_dir, T)
        # share of every batch taken from approved user lessons; higher for learning runs
        mix = float(cfg.get("lesson_mix", 0.5 if auto else 0.25))

        def batch(src, n=None, starts=None):
            n = n or B
            ix = r.integers(0, len(src) - T - 1, n)
            if starts is not None:  # half the lesson windows start exactly where a lesson starts
                aligned = r.random(n) < 0.5
                ix[aligned] = r.choice(starts, int(aligned.sum()))
            x = np.stack([src[i:i + T] for i in ix])
            y = np.stack([src[i + 1:i + T + 1] for i in ix])
            return xp.asarray(x), xp.asarray(y)
        n_classes = 0
        classes, old_classes = None, []
    else:
        S = int(cfg["image_size"])
        X, y, classes, user_mask = D.load_images(ds_dir, S, with_user=True)
        if kind == "vision" and len(classes) < 2:
            raise ValueError("Image analysis models need at least 2 classes (folders).")
        old_classes = meta.get("classes") or []
        if old_classes and not set(old_classes) <= set(classes):
            raise ValueError("Some classes this model knows were removed from the dataset. Create a new model.")
        meta["classes"] = classes
        n_classes = len(classes)
        perm = np.random.default_rng(7).permutation(len(X))
        n_val = max(1, len(X) // 10) if len(X) >= 20 else 0
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        user_idx = np.array([i for i in tr_idx if user_mask[i]])
        umix = 0.5 if auto else 0.25  # pictures users taught get extra attention
        Xf = D.to_float(X)

        def batch(idx_pool, augment=True):
            ix = r.choice(idx_pool, min(B, len(idx_pool)), replace=len(idx_pool) < B)
            if idx_pool is tr_idx and len(user_idx):
                nu = max(1, int(round(len(ix) * umix)))
                ix[:nu] = r.choice(user_idx, nu)
            xb = Xf[ix].copy()
            if augment:
                flip = r.random(len(ix)) < 0.5
                xb[flip] = xb[flip][:, :, ::-1, :]
            return Tensor(xp.asarray(xb)), y[ix]

    # ----------------------------------------------------------- model
    model = build_model(kind, cfg, n_classes)
    opt = AdamW(model.named_parameters(), lr=float(cfg["lr"]),
                weight_decay=float(cfg.get("weight_decay", 0.01)))
    sd = load_weights(mid)
    if sd is None and cfg.get("init_from"):
        meta["started_from"] = warm_start(kind, model, int(cfg["init_from"]), classes if kind != "chat" else None, cfg)
    elif sd is not None:
        if kind != "chat" and old_classes and old_classes != classes:
            sd = remap_classes(kind, sd, old_classes, classes, cfg)
            meta["grew_classes"] = [c for c in classes if c not in old_classes]
        model.load_state_dict(sd)
    if sd is not None:
        op = os.path.join(mdir, "optim.npz")
        if os.path.exists(op):
            with np.load(op) as z:
                opt.load_state_dict({k: z[k] for k in z.files})

    teacher = cmap = None
    if cfg.get("teacher") and kind in ("chat", "vision"):
        teacher, cmap, tname = load_teacher(kind, int(cfg["teacher"]), classes, cfg)
        meta["teacher"] = tname
    alpha, ttemp = float(cfg.get("distill_alpha", 0.5)), float(cfg.get("distill_temp", 2.0))

    start = int(row["step"])
    target = start + extra_steps
    backup_checkpoint(mid, start, meta_before)
    base_lr = float(cfg["lr"]) * lr_scale
    baseline = _evaluate(kind, model, locals()) if (auto and sd is not None) else None
    lessons_before = _lesson_loss(model, locals()) if (auto and sd is not None and kind == "chat") else None
    eval_every = int(cfg.get("eval_interval") or max(10, min(200, extra_steps // 25)))
    warmup = int(cfg.get("warmup") or min(100, max(1, extra_steps // 10)))
    clip = float(cfg.get("grad_clip", 1.0))
    _update(mid, status="training", error=None, target_steps=target, params=model.num_params(),
            meta=json.dumps(meta))

    running, n_run, t0 = 0.0, 0, time.time()
    share = ThreadShare(mid)
    share.check()
    step = start
    for step in range(start, target):
        if os.path.exists(stop_flag(mid)):
            break
        if step % 10 == 0:
            share.check()
        lr = cosine_lr(step - start, extra_steps, base_lr, warmup=warmup)
        opt.zero_grad()
        if kind == "chat":
            if lesson_tok is not None:
                nl = max(1, int(round(B * mix)))
                x1, y1 = batch(train_tok, B - nl)
                x2, y2 = batch(lesson_tok, nl, lesson_starts)
                xb, yb = xp.concatenate([x1, x2]), xp.concatenate([y1, y2])
            else:
                xb, yb = batch(train_tok)
            if teacher is None:
                loss = model.loss(xb, yb)
            else:  # learn from the data and from the teacher's full answer at every position
                logits = model.forward(xb)
                V = logits.shape[-1]
                flat = logits.reshape(-1, V)
                loss = cross_entropy(flat, yb.reshape(-1)) * (1 - alpha) + \
                    soft_cross_entropy(flat * (1.0 / ttemp), teacher_probs(kind, teacher, None, xb, None, ttemp)) * (alpha * ttemp * ttemp)
        elif kind == "vision":
            xb, yb = batch(tr_idx)
            if teacher is None:
                loss = model.loss(xb, yb)
            else:
                logits = model.forward(xb)
                loss = cross_entropy(logits, yb) * (1 - alpha) + \
                    soft_cross_entropy(logits * (1.0 / ttemp), teacher_probs(kind, teacher, cmap, xb, yb, ttemp)) * (alpha * ttemp * ttemp)
        else:
            xb, yb = batch(tr_idx, augment=True)
            loss, _, _ = model.loss(xb, yb)
        loss.backward()
        opt.clip_grad_norm(clip)
        opt.step(lr)
        lv = loss.item()
        if not np.isfinite(lv):
            raise FloatingPointError("Loss became NaN/inf. Lower the learning rate and retrain.")
        running += lv
        n_run += 1

        done = step + 1
        if done % eval_every == 0 or done == target:
            metric = _evaluate(kind, model, locals())
            meta["steps_per_sec"] = round(n_run / max(1e-6, time.time() - t0), 2)
            meta["threads"] = share.current
            meta.update(_preview(kind, model, mid, meta))
            save_checkpoint(mid, model, opt)
            db.ex("INSERT INTO training_log(model_id,step,loss,metric,created_at) VALUES(?,?,?,?,?)",
                  (mid, done, running / n_run, metric, db.now()))
            _update(mid, step=done, meta=json.dumps(meta))
            running, n_run, t0 = 0.0, 0, time.time()

    stopped = os.path.exists(stop_flag(mid))
    if n_run:  # save partial interval
        save_checkpoint(mid, model, opt)
        done = step + (0 if stopped else 1)
        _update(mid, step=done)
    if stopped:
        os.remove(stop_flag(mid))

    if auto and baseline is not None and not stopped:
        after = _evaluate(kind, model, locals())
        result = {"at": time.time(), "before": baseline, "after": after, "steps": extra_steps}
        if lessons_before is not None:
            result["lessons_before"], result["lessons_after"] = lessons_before, _lesson_loss(model, locals())
        if _worse(kind, baseline, after):
            restore_checkpoint(mid)  # the lesson made things worse: undo it
            meta_before["last_learn"] = dict(result, outcome="reverted")
            _update(mid, step=start, meta=json.dumps(meta_before), status="ready")
            return
        meta["last_learn"] = dict(result, outcome="kept")
        _update(mid, meta=json.dumps(meta))
    _update(mid, status="stopped" if stopped else "ready")


def _lesson_loss(model, env):
    """How well the model knows its approved lessons (lower is better)."""
    tok, starts = env.get("lesson_tok"), env.get("lesson_starts")
    if tok is None:
        return None
    T = int(env["cfg"]["block_size"])
    ix = starts[:16]
    x = np.stack([tok[i:i + T] for i in ix])
    y = np.stack([tok[i + 1:i + T + 1] for i in ix])
    with no_grad():
        return model.loss(xp.asarray(x), xp.asarray(y)).item()


def _evaluate(kind, model, env):
    with no_grad():
        if kind == "chat":  # fixed positions so scores are comparable between evaluations
            val, T = env["val_tok"], int(env["cfg"]["block_size"])
            rr = np.random.default_rng(0)
            losses = []
            for _ in range(4):
                ix = rr.integers(0, len(val) - T - 1, int(env["B"]))
                x = np.stack([val[i:i + T] for i in ix])
                y = np.stack([val[i + 1:i + T + 1] for i in ix])
                losses.append(model.loss(xp.asarray(x), xp.asarray(y)).item())
            return float(np.mean(losses))  # validation loss
        val_idx = env["val_idx"]
        if len(val_idx) == 0:
            return None
        if kind == "vision":
            correct = 0
            for i in range(0, len(val_idx), 64):
                ix = val_idx[i:i + 64]
                p = model.predict(Tensor(xp.asarray(env["Xf"][ix])))
                correct += int((p.argmax(1) == env["y"][ix]).sum())
            return correct / len(val_idx)  # validation accuracy
        ix = val_idx[:64]
        from .engine import nn as NN
        saved = NN._rng
        NN.seed(0)  # fixed sampling noise so scores are comparable
        try:
            _, recon, _ = model.loss(Tensor(xp.asarray(env["Xf"][ix])), env["y"][ix])
        finally:
            NN._rng = saved
        return recon.item()  # validation reconstruction loss


def _preview(kind, model, mid, meta):
    if kind == "chat":
        txt = model.generate(b"User: Hello!\nDeer:", max_new=300, temperature=0.8, seed=0,
                             stop=[bytes([3]), b"\nUser:"])
        return {"preview_text": txt}
    if kind == "canvas":
        k = len(meta["classes"])
        ids = [i % k for i in range(8)]
        grid_png(model.sample(ids, seed=0), os.path.join(db.model_dir(mid), "preview.png"))
        return {"preview_image": True}
    return {}
