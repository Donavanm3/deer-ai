"""Layers and module system for Deer AI, built only on deer.engine.tensor."""
import numpy as np
from .tensor import Tensor, xp, DTYPE, to_numpy, layer_norm, conv2d, embedding

_rng = np.random.default_rng()


def seed(n):
    global _rng
    _rng = np.random.default_rng(n)


def rng():
    return _rng


def Parameter(arr):
    return Tensor(xp.asarray(arr, dtype=DTYPE), requires_grad=True)


class Module:
    training = True

    def named_parameters(self, prefix=""):
        for k, v in self.__dict__.items():
            name = f"{prefix}{k}"
            if isinstance(v, Tensor) and v.requires_grad:
                yield name, v
            elif isinstance(v, Module):
                yield from v.named_parameters(name + ".")
            elif isinstance(v, (list, tuple)):
                for i, m in enumerate(v):
                    if isinstance(m, Module):
                        yield from m.named_parameters(f"{name}.{i}.")

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def num_params(self):
        return int(sum(p.data.size for p in self.parameters()))

    def state_dict(self):
        return {k: to_numpy(v.data) for k, v in self.named_parameters()}

    def load_state_dict(self, sd):
        params = dict(self.named_parameters())
        missing = [k for k in params if k not in sd]
        if missing:
            raise ValueError(f"checkpoint missing parameters: {missing[:5]}")
        for k, p in params.items():
            arr = sd[k]
            if tuple(arr.shape) != tuple(p.shape):
                raise ValueError(f"shape mismatch for {k}: {arr.shape} vs {p.shape}")
            p.data = xp.asarray(arr, dtype=DTYPE)

    def __call__(self, *a, **kw):
        return self.forward(*a, **kw)


class Linear(Module):
    def __init__(self, n_in, n_out, bias=True, std=None):
        std = std if std is not None else (2.0 / n_in) ** 0.5
        self.w = Parameter(_rng.normal(0, std, (n_in, n_out)))
        self.b = Parameter(np.zeros(n_out)) if bias else None

    def forward(self, x):
        y = x @ self.w
        return y + self.b if self.b is not None else y


class Embedding(Module):
    def __init__(self, n, dim, std=0.02):
        self.w = Parameter(_rng.normal(0, std, (n, dim)))

    def forward(self, idx):
        return embedding(self.w, idx)


class LayerNorm(Module):
    def __init__(self, dim):
        self.g = Parameter(np.ones(dim))
        self.b = Parameter(np.zeros(dim))

    def forward(self, x):
        return layer_norm(x, self.g, self.b)


class Conv2d(Module):
    def __init__(self, c_in, c_out, k=3, stride=1, pad=1):
        fan_in = c_in * k * k
        self.w = Parameter(_rng.normal(0, (2.0 / fan_in) ** 0.5, (k, k, c_in, c_out)))
        self.b = Parameter(np.zeros(c_out))
        self.stride, self.pad = stride, pad

    def forward(self, x):
        return conv2d(x, self.w, self.b, self.stride, self.pad)
