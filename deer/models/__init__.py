"""
Deer AI model architectures, all implemented from scratch on deer.engine:

  chat   -> DeerChat   : byte-level GPT-style transformer (text generation / chat)
  vision -> DeerVision : convolutional image classifier (image analysis)
  canvas -> DeerCanvas : conditional variational autoencoder (image generation)
"""
import copy
import numpy as np
from ..engine import Tensor, no_grad, xp, concat, cross_entropy, bce_with_logits_sum, upsample2x, to_numpy
from ..engine.nn import Module, Linear, Embedding, LayerNorm, Conv2d, rng

# --------------------------------------------------------------------- presets
PRESETS = {
    "chat": {
        "tiny":   {"n_layer": 2, "n_head": 2, "n_embd": 64,  "block_size": 64,  "batch_size": 16, "lr": 3e-3},
        "small":  {"n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 128, "batch_size": 16, "lr": 1.5e-3},
        "medium": {"n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 256, "batch_size": 12, "lr": 1e-3},
        # coding: code needs to see further back, so these read longer context
        "code-small":  {"n_layer": 4, "n_head": 4, "n_embd": 160, "block_size": 256, "batch_size": 12, "lr": 1.2e-3},
        "code-medium": {"n_layer": 6, "n_head": 6, "n_embd": 240, "block_size": 384, "batch_size": 8, "lr": 8e-4},
    },
    "vision": {
        "tiny":   {"image_size": 32, "channels": [16, 32, 64],  "batch_size": 32, "lr": 2e-3},
        "small":  {"image_size": 32, "channels": [32, 64, 128], "batch_size": 32, "lr": 1.5e-3},
        "medium": {"image_size": 64, "channels": [32, 64, 128], "batch_size": 24, "lr": 1e-3},
    },
    "canvas": {
        "tiny":   {"image_size": 32, "latent": 16, "hidden": 128, "batch_size": 32, "lr": 2e-3, "beta": 1.0},
        "small":  {"image_size": 32, "latent": 32, "hidden": 256, "batch_size": 32, "lr": 1e-3, "beta": 1.0},
        "medium": {"image_size": 64, "latent": 64, "hidden": 384, "batch_size": 16, "lr": 8e-4, "beta": 1.0},
    },
}
KIND_DATASET = {"chat": "text", "vision": "images", "canvas": "images"}

LIMITS = {
    "n_layer": (1, 24), "n_head": (1, 16), "n_embd": (16, 1024), "block_size": (16, 1024),
    "batch_size": (1, 256), "image_size": (16, 128), "latent": (2, 512), "hidden": (16, 2048),
}


def build_config(kind, preset="small", overrides=None):
    if kind not in PRESETS:
        raise ValueError(f"unknown model kind '{kind}'")
    if preset not in PRESETS[kind]:
        raise ValueError(f"unknown preset '{preset}'")
    cfg = copy.deepcopy(PRESETS[kind][preset])
    for k, v in (overrides or {}).items():
        if k not in cfg and k not in ("weight_decay", "grad_clip", "warmup", "eval_interval",
                                      "init_from", "teacher", "distill_alpha", "distill_temp"):
            raise ValueError(f"unknown config key '{k}' for {kind}")
        cfg[k] = v
    for k, (lo, hi) in LIMITS.items():
        if k in cfg and not (lo <= int(cfg[k]) <= hi):
            raise ValueError(f"{k} must be between {lo} and {hi}")
    if kind == "chat" and cfg["n_embd"] % cfg["n_head"]:
        raise ValueError("n_embd must be divisible by n_head")
    if kind in ("vision", "canvas") and cfg["image_size"] % 4:
        raise ValueError("image_size must be divisible by 4")
    if not (0 < float(cfg["lr"]) < 1):
        raise ValueError("lr must be between 0 and 1")
    if "distill_alpha" in cfg and not (0 < float(cfg["distill_alpha"]) < 1):
        raise ValueError("distill_alpha must be between 0 and 1")
    if "distill_temp" in cfg and not (0.5 <= float(cfg["distill_temp"]) <= 10):
        raise ValueError("distill_temp must be between 0.5 and 10")
    return cfg


# ======================================================================= chat
class Block(Module):
    def __init__(self, C, n_head, n_layer):
        std = 0.02
        self.ln1 = LayerNorm(C)
        self.qkv = Linear(C, 3 * C, std=std)
        self.proj = Linear(C, C, std=std / (2 * n_layer) ** 0.5)
        self.ln2 = LayerNorm(C)
        self.fc = Linear(C, 4 * C, std=std)
        self.out = Linear(4 * C, C, std=std / (2 * n_layer) ** 0.5)
        self.n_head = n_head

    def forward(self, x, mask):
        B, T, C = x.shape
        h, d = self.n_head, C // self.n_head
        qkv = self.qkv(self.ln1(x)).reshape(B, T, 3, h, d).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / d ** 0.5)
        att = att.masked_fill(mask, -1e9).softmax(-1)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        x = x + self.proj(y)
        x = x + self.out(self.fc(self.ln2(x)).gelu())
        return x


class DeerChat(Module):
    VOCAB = 256  # raw bytes: no external tokenizer needed

    def __init__(self, cfg):
        C = cfg["n_embd"]
        self.cfg = cfg
        self.tok = Embedding(self.VOCAB, C)
        self.pos = Embedding(cfg["block_size"], C)
        self.blocks = [Block(C, cfg["n_head"], cfg["n_layer"]) for _ in range(cfg["n_layer"])]
        self.ln = LayerNorm(C)
        self.head = Linear(C, self.VOCAB, bias=False, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        mask = xp.triu(xp.ones((T, T), dtype=bool), 1)
        x = self.tok(idx) + self.pos(xp.arange(T))
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))

    def loss(self, idx, targets):
        logits = self.forward(idx)
        B, T, V = logits.shape
        return cross_entropy(logits.reshape(B * T, V), targets.reshape(-1))

    # ---------------------------------------------------------- fast inference
    def stream(self, prompt_bytes, max_new=240, temperature=0.8, top_k=40, stop=None, seed=None, deadline=None):
        """Yields decoded text as it's generated. Uses a key/value cache, so each new character
        costs one small step instead of re-reading the whole context."""
        import codecs
        import time as _time
        stops = [stop] if isinstance(stop, (bytes, bytearray)) else list(stop or [])
        longest = max((len(x) for x in stops), default=0)
        W = {k: to_numpy(v.data).astype(np.float32) for k, v in self.named_parameters()}
        cfg = self.cfg
        L, H, C, bs = cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["block_size"]
        d = C // H
        scale = np.float32(1.0 / np.sqrt(d))
        r = np.random.default_rng(seed)
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def ln(x, p):
            mu = x.mean(-1, keepdims=True)
            var = x.var(-1, keepdims=True)
            return (x - mu) / np.sqrt(var + 1e-5) * W[p + ".g"] + W[p + ".b"]

        def gelu(x):
            return 0.5 * x * (1 + np.tanh(np.float32(0.7978845608) * (x + np.float32(0.044715) * (x * x * x))))

        def softmax(a):
            a = a - a.max(-1, keepdims=True)
            e = np.exp(a)
            return e / e.sum(-1, keepdims=True)

        def prime(seq):
            T = len(seq)
            x = W["tok.w"][seq] + W["pos.w"][:T]
            mask = np.triu(np.ones((T, T), bool), 1)
            Ks, Vs = [], []
            for l in range(L):
                p = f"blocks.{l}."
                qkv = (ln(x, p + "ln1") @ W[p + "qkv.w"] + W[p + "qkv.b"]).reshape(T, 3, H, d).transpose(1, 2, 0, 3)
                q, k, v = qkv[0], qkv[1], qkv[2]
                att = (q @ k.transpose(0, 2, 1)) * scale
                att[:, mask] = -1e9
                y = (softmax(att) @ v).transpose(1, 0, 2).reshape(T, C)
                x = x + y @ W[p + "proj.w"] + W[p + "proj.b"]
                x = x + gelu(ln(x, p + "ln2") @ W[p + "fc.w"] + W[p + "fc.b"]) @ W[p + "out.w"] + W[p + "out.b"]
                Ks.append(k)
                Vs.append(v)
            return Ks, Vs, ln(x[-1], "ln") @ W["head.w"]

        def step(tok, pos, Ks, Vs):
            x = W["tok.w"][tok] + W["pos.w"][pos]
            for l in range(L):
                p = f"blocks.{l}."
                qkv = (ln(x, p + "ln1") @ W[p + "qkv.w"] + W[p + "qkv.b"]).reshape(3, H, d)
                q, k, v = qkv[0], qkv[1], qkv[2]
                Ks[l] = np.concatenate([Ks[l], k[:, None, :]], axis=1)
                Vs[l] = np.concatenate([Vs[l], v[:, None, :]], axis=1)
                att = softmax((Ks[l] @ q[:, :, None])[:, :, 0] * scale)          # (H, T)
                y = (att[:, None, :] @ Vs[l])[:, 0, :].reshape(C)
                x = x + y @ W[p + "proj.w"] + W[p + "proj.b"]
                x = x + gelu(ln(x, p + "ln2") @ W[p + "fc.w"] + W[p + "fc.b"]) @ W[p + "out.w"] + W[p + "out.b"]
            return ln(x, "ln") @ W["head.w"]

        ids = list(prompt_bytes)[-(bs - 1):] or [10]
        Ks, Vs, logits = prime(ids)
        pending = bytearray()
        for _ in range(max_new):
            if deadline is not None and _time.monotonic() > deadline:
                break
            lg = logits.astype(np.float64) / max(temperature, 1e-4)
            if top_k and top_k < len(lg):
                cut = np.partition(lg, -top_k)[-top_k]
                lg[lg < cut] = -np.inf
            pr = np.exp(lg - lg.max())
            pr /= pr.sum()
            nxt = int(r.choice(len(pr), p=pr))
            pending.append(nxt)
            hit = next((x for x in stops if pending.endswith(x)), None)
            if hit is not None:
                del pending[len(pending) - len(hit):]
                piece = dec.decode(bytes(pending))
                if piece:
                    yield piece
                pending.clear()
                break
            # hold back only bytes that could still turn into a stop sequence
            keep = 0
            for k in range(1, min(longest, len(pending)) + 1):
                if any(x.startswith(bytes(pending[-k:])) for x in stops):
                    keep = k
            if len(pending) > keep:
                piece = dec.decode(bytes(pending[:len(pending) - keep]))
                del pending[:len(pending) - keep]
                if piece:
                    yield piece
            ids.append(nxt)
            if len(ids) >= bs:  # context full: slide the window and rebuild the cache
                ids = ids[-(bs // 2):]
                Ks, Vs, logits = prime(ids)
            else:
                logits = step(nxt, len(ids) - 1, Ks, Vs)
        tail = dec.decode(bytes(pending), final=True)
        if tail:
            yield tail

    def generate(self, prompt_bytes, max_new=200, temperature=0.8, top_k=40, stop=None, seed=None, deadline=None):
        return "".join(self.stream(prompt_bytes, max_new, temperature, top_k, stop, seed, deadline))


# ===================================================================== vision
class DeerVision(Module):
    def __init__(self, cfg, n_classes):
        c1, c2, c3 = cfg["channels"]
        self.cfg = cfg
        self.convs = [
            Conv2d(3, c1, 3, 1, 1), Conv2d(c1, c1, 3, 2, 1),
            Conv2d(c1, c2, 3, 1, 1), Conv2d(c2, c2, 3, 2, 1),
            Conv2d(c2, c3, 3, 1, 1), Conv2d(c3, c3, 3, 2, 1),
        ]
        self.fc = Linear(c3, n_classes)

    def forward(self, x):
        for c in self.convs:
            x = c(x).relu()
        return self.fc(x.mean(axis=(1, 2)))

    def loss(self, x, y):
        return cross_entropy(self.forward(x), y)

    def predict(self, x):
        with no_grad():
            return to_numpy(self.forward(x).softmax(-1).data)


# ===================================================================== canvas
class DeerCanvas(Module):
    def __init__(self, cfg, n_classes):
        S, Z, H = cfg["image_size"], cfg["latent"], cfg["hidden"]
        self.cfg, self.K, self.S = cfg, n_classes, S
        s4 = S // 4
        self.flat = 64 * s4 * s4
        # encoder
        self.e1 = Conv2d(3, 32, 3, 2, 1)
        self.e2 = Conv2d(32, 64, 3, 2, 1)
        self.eh = Linear(self.flat + n_classes, H)
        self.mu = Linear(H, Z, std=0.01)
        self.lv = Linear(H, Z, std=0.01)
        # decoder
        self.dh = Linear(Z + n_classes, H)
        self.dfc = Linear(H, self.flat)
        self.d1 = Conv2d(64, 64, 3, 1, 1)
        self.d2 = Conv2d(64, 32, 3, 1, 1)
        self.d3 = Conv2d(32, 3, 3, 1, 1)

    def onehot(self, y):
        oh = np.zeros((len(y), self.K), dtype=np.float32)
        oh[np.arange(len(y)), np.asarray(y)] = 1
        return Tensor(xp.asarray(oh))

    def decode(self, z, oh):
        s4 = self.S // 4
        h = self.dfc(self.dh(concat([z, oh], axis=1)).relu()).relu()
        h = h.reshape(h.shape[0], s4, s4, 64)
        h = self.d1(upsample2x(h)).relu()
        h = self.d2(upsample2x(h)).relu()
        return self.d3(h)  # logits

    def loss(self, x, y):
        N = x.shape[0]
        oh = self.onehot(y)
        h = self.e2(self.e1(x).relu()).relu().reshape(N, self.flat)
        h = self.eh(concat([h, oh], axis=1)).relu()
        mu, lv = self.mu(h), self.lv(h)
        eps = Tensor(xp.asarray(rng().normal(0, 1, mu.shape), dtype=np.float32))
        z = mu + (lv * 0.5).exp() * eps
        recon = bce_with_logits_sum(self.decode(z, oh), x, N)
        kl = ((mu * mu) + lv.exp() - lv - 1.0).sum() * (0.5 / N)
        return recon + kl * float(self.cfg.get("beta", 1.0)), recon, kl

    def sample(self, class_ids, temperature=1.0, seed=None):
        r = np.random.default_rng(seed)
        z = r.normal(0, temperature, (len(class_ids), self.cfg["latent"])).astype(np.float32)
        with no_grad():
            logits = self.decode(Tensor(xp.asarray(z)), self.onehot(class_ids))
            return to_numpy(logits.sigmoid().data)  # (N,S,S,3) in [0,1]


def build_model(kind, cfg, n_classes=0):
    if kind == "chat":
        return DeerChat(cfg)
    if kind == "vision":
        return DeerVision(cfg, n_classes)
    if kind == "canvas":
        return DeerCanvas(cfg, n_classes)
    raise ValueError(kind)
