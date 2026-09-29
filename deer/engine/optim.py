"""AdamW optimizer and gradient clipping, written from scratch."""
import math
import numpy as np
from .tensor import xp, DTYPE, to_numpy


class AdamW:
    def __init__(self, named_params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01):
        self.named = list(named_params)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.t = 0
        self.m = {n: xp.zeros_like(p.data) for n, p in self.named}
        self.v = {n: xp.zeros_like(p.data) for n, p in self.named}

    def zero_grad(self):
        for _, p in self.named:
            p.grad = None

    def clip_grad_norm(self, max_norm):
        total = 0.0
        for _, p in self.named:
            if p.grad is not None:
                total += float((p.grad * p.grad).sum())
        norm = math.sqrt(total)
        if max_norm and norm > max_norm:
            scale = max_norm / (norm + 1e-6)
            for _, p in self.named:
                if p.grad is not None:
                    p.grad *= scale
        return norm

    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for n, p in self.named:
            g = p.grad
            if g is None:
                continue
            m, v = self.m[n], self.v[n]
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * (g * g)
            if self.wd and p.data.ndim >= 2:
                p.data *= (1 - lr * self.wd)
            p.data -= (lr * (m / bc1) / (xp.sqrt(v / bc2) + self.eps)).astype(DTYPE)

    def state_dict(self):
        sd = {"__t": np.array(self.t)}
        for n in self.m:
            sd[f"m.{n}"] = to_numpy(self.m[n])
            sd[f"v.{n}"] = to_numpy(self.v[n])
        return sd

    def load_state_dict(self, sd):
        if "__t" not in sd:
            return
        self.t = int(sd["__t"])
        for n in self.m:
            if f"m.{n}" in sd and sd[f"m.{n}"].shape == self.m[n].shape:
                self.m[n] = xp.asarray(sd[f"m.{n}"], dtype=DTYPE)
                self.v[n] = xp.asarray(sd[f"v.{n}"], dtype=DTYPE)


def cosine_lr(step, total, base, warmup=100, floor=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog)))
