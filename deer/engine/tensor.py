"""
DeerTensor: a from-scratch reverse-mode automatic differentiation engine.

No machine-learning frameworks are used. Arrays are plain NumPy (or CuPy when
DEER_DEVICE=gpu and CuPy is installed - experimental). Every operation below
defines its own forward math and its own hand-written gradient.
"""
import os
import contextlib
import threading
import numpy as np

xp = np
if os.environ.get("DEER_DEVICE", "cpu").lower() == "gpu":
    try:  # optional GPU backend; same array API as NumPy
        import cupy as _cp  # type: ignore
        xp = _cp
    except Exception:  # pragma: no cover
        xp = np

DTYPE = np.float32
_state = threading.local()  # grad mode is per-thread so serving and training can share a process


def grad_enabled():
    return getattr(_state, "grad", True)


@contextlib.contextmanager
def no_grad():
    prev = grad_enabled()
    _state.grad = False
    try:
        yield
    finally:
        _state.grad = prev


def to_numpy(a):
    if xp is np:
        return np.asarray(a)
    return xp.asnumpy(a)  # pragma: no cover


def asarray(a, dtype=DTYPE):
    if isinstance(a, Tensor):
        return a.data
    return xp.asarray(a, dtype=dtype)


def _unbroadcast(g, shape):
    """Sum a gradient back down to `shape` after NumPy broadcasting."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def _scatter_add(target, idx, values):
    if hasattr(xp, "add") and hasattr(xp.add, "at"):
        xp.add.at(target, idx, values)
    else:  # pragma: no cover
        import cupyx  # type: ignore
        cupyx.scatter_add(target, idx, values)


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward", "name")

    def __init__(self, data, requires_grad=False, _parents=(), _backward=None, name=None):
        self.data = data if (hasattr(data, "dtype") and not isinstance(data, np.generic)) else xp.asarray(data, dtype=DTYPE)
        self.grad = None
        self.requires_grad = requires_grad
        self._parents = _parents
        self._backward = _backward
        self.name = name

    # ---------------------------------------------------------------- basics
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def numpy(self):
        return to_numpy(self.data)

    def item(self):
        return float(to_numpy(self.data).reshape(-1)[0])

    def detach(self):
        return Tensor(self.data)

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    # ------------------------------------------------------------- autograd
    @staticmethod
    def _make(data, parents, backward):
        needs = grad_enabled() and any(p.requires_grad for p in parents)
        if not needs:
            return Tensor(data)
        return Tensor(data, requires_grad=True, _parents=parents, _backward=backward)

    def backward(self, grad=None):
        if grad is None:
            grad = xp.ones_like(self.data)
        # iterative topological sort (safe for deep graphs)
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if p.requires_grad and id(p) not in seen:
                    stack.append((p, False))
        grads = {id(self): grad}
        for node in reversed(order):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._backward is None:  # leaf
                node.grad = g if node.grad is None else node.grad + g
                continue
            pgrads = node._backward(g)
            for p, pg in zip(node._parents, pgrads):
                if pg is None or not p.requires_grad:
                    continue
                if id(p) in grads:
                    grads[id(p)] = grads[id(p)] + pg
                else:
                    grads[id(p)] = pg
            # free graph references once used
            node._parents = ()
            node._backward = None

    # ----------------------------------------------------------- arithmetic
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(asarray(other))
        a, b = self, other
        return Tensor._make(a.data + b.data, (a, b),
                            lambda g: (_unbroadcast(g, a.shape), _unbroadcast(g, b.shape)))

    __radd__ = __add__

    def __neg__(self):
        return Tensor._make(-self.data, (self,), lambda g: (-g,))

    def __sub__(self, other):
        return self + (-(other if isinstance(other, Tensor) else Tensor(asarray(other))))

    def __rsub__(self, other):
        return Tensor(asarray(other)) + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(asarray(other))
        a, b = self, other
        return Tensor._make(a.data * b.data, (a, b),
                            lambda g: (_unbroadcast(g * b.data, a.shape), _unbroadcast(g * a.data, b.shape)))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(asarray(other))
        a, b = self, other
        return Tensor._make(a.data / b.data, (a, b),
                            lambda g: (_unbroadcast(g / b.data, a.shape),
                                       _unbroadcast(-g * a.data / (b.data * b.data), b.shape)))

    def __pow__(self, p):
        assert isinstance(p, (int, float))
        a = self
        return Tensor._make(a.data ** p, (a,), lambda g: (g * p * a.data ** (p - 1),))

    def __matmul__(self, other):
        a, b = self, other

        def bw(g):
            ga = g @ xp.swapaxes(b.data, -1, -2)
            gb = xp.swapaxes(a.data, -1, -2) @ g
            return _unbroadcast(ga, a.shape), _unbroadcast(gb, b.shape)
        return Tensor._make(a.data @ b.data, (a, b), bw)

    # ------------------------------------------------------------ reductions
    def sum(self, axis=None, keepdims=False):
        a = self

        def bw(g):
            if axis is not None and not keepdims:
                g = xp.expand_dims(g, axis)
            return (xp.broadcast_to(g, a.shape).astype(DTYPE),)
        return Tensor._make(a.data.sum(axis=axis, keepdims=keepdims), (a,), bw)

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            n = 1
            for ax in axes:
                n *= self.data.shape[ax]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # --------------------------------------------------------------- shapes
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        a = self
        return Tensor._make(a.data.reshape(shape), (a,), lambda g: (g.reshape(a.shape),))

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        inv = tuple(int(i) for i in np.argsort(axes))
        a = self
        return Tensor._make(a.data.transpose(axes), (a,), lambda g: (g.transpose(inv),))

    def __getitem__(self, idx):
        a = self

        def bw(g):
            full = xp.zeros_like(a.data)
            full[idx] += g
            return (full,)
        return Tensor._make(a.data[idx], (a,), bw)

    # --------------------------------------------------------- elementwise
    def exp(self):
        a = self
        out = xp.exp(a.data)
        return Tensor._make(out, (a,), lambda g: (g * out,))

    def log(self):
        a = self
        return Tensor._make(xp.log(a.data), (a,), lambda g: (g / a.data,))

    def relu(self):
        a = self
        mask = a.data > 0
        return Tensor._make(a.data * mask, (a,), lambda g: (g * mask,))

    def tanh(self):
        a = self
        out = xp.tanh(a.data)
        return Tensor._make(out, (a,), lambda g: (g * (1 - out * out),))

    def sigmoid(self):
        a = self
        out = 1.0 / (1.0 + xp.exp(-a.data))
        return Tensor._make(out, (a,), lambda g: (g * out * (1 - out),))

    def gelu(self):
        a = self
        x = a.data
        c = DTYPE(0.7978845608)  # sqrt(2/pi)
        inner = c * (x + DTYPE(0.044715) * (x * x * x))  # x*x*x: much faster than x**3 in NumPy
        t = xp.tanh(inner)
        out = 0.5 * x * (1 + t)

        def bw(g):
            dinner = c * (1 + 3 * DTYPE(0.044715) * x * x)
            d = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dinner
            return (g * d,)
        return Tensor._make(out, (a,), bw)

    def softmax(self, axis=-1):
        a = self
        z = a.data - a.data.max(axis=axis, keepdims=True)
        e = xp.exp(z)
        out = e / e.sum(axis=axis, keepdims=True)
        return Tensor._make(out, (a,), lambda g: (out * (g - (g * out).sum(axis=axis, keepdims=True)),))

    def masked_fill(self, mask, value):
        """mask: boolean array broadcastable to self; masked positions become `value`."""
        a = self
        out = xp.where(mask, DTYPE(value), a.data)
        return Tensor._make(out, (a,), lambda g: (xp.where(mask, DTYPE(0), g),))


# ============================================================ free functions

def concat(tensors, axis=-1):
    datas = [t.data for t in tensors]
    out = xp.concatenate(datas, axis=axis)
    sizes = [d.shape[axis] for d in datas]

    def bw(g):
        grads, start = [], 0
        for s in sizes:
            sl = [slice(None)] * g.ndim
            sl[axis] = slice(start, start + s)
            grads.append(g[tuple(sl)])
            start += s
        return tuple(grads)
    return Tensor._make(out, tuple(tensors), bw)


def embedding(weight, idx):
    """weight: (V, C) Tensor, idx: int array of any shape -> (*idx.shape, C)."""
    idx = xp.asarray(idx)

    def bw(g):
        full = xp.zeros_like(weight.data)
        _scatter_add(full, idx.reshape(-1), g.reshape(-1, weight.shape[1]))
        return (full,)
    return Tensor._make(weight.data[idx], (weight,), bw)


def cross_entropy(logits, targets):
    """Mean cross-entropy. logits (N, V), targets int array (N,)."""
    targets = xp.asarray(targets)
    x = logits.data
    z = x - x.max(axis=1, keepdims=True)
    e = xp.exp(z)
    s = e.sum(axis=1, keepdims=True)
    logp = z - xp.log(s)
    n = x.shape[0]
    loss = -logp[xp.arange(n), targets].mean()

    def bw(g):
        p = e / s
        p[xp.arange(n), targets] -= 1
        return (p * (g / n),)
    return Tensor._make(xp.asarray(loss, dtype=DTYPE), (logits,), bw)


def soft_cross_entropy(logits, target_probs):
    """Mean cross-entropy against a full probability distribution (a teacher's answer)
    instead of one right answer. logits (N, V), target_probs (N, V) rows summing to 1."""
    p = asarray(target_probs)
    x = logits.data
    z = x - x.max(axis=1, keepdims=True)
    e = xp.exp(z)
    s = e.sum(axis=1, keepdims=True)
    logq = z - xp.log(s)
    n = x.shape[0]
    loss = -(p * logq).sum() / n

    def bw(g):
        return ((e / s - p) * (g / n),)
    return Tensor._make(xp.asarray(loss, dtype=DTYPE), (logits,), bw)


def bce_with_logits_sum(logits, target, batch):
    """Binary cross-entropy summed over elements, divided by batch size."""
    t = asarray(target)
    x = logits.data
    val = (xp.maximum(x, 0) - x * t + xp.log1p(xp.exp(-xp.abs(x)))).sum() / batch

    def bw(g):
        sig = 1.0 / (1.0 + xp.exp(-x))
        return ((sig - t) * (g / batch),)
    return Tensor._make(xp.asarray(val, dtype=DTYPE), (logits,), bw)


def layer_norm(x, gamma, beta, eps=1e-5):
    xd = x.data
    mu = xd.mean(axis=-1, keepdims=True)
    var = xd.var(axis=-1, keepdims=True)
    rstd = 1.0 / xp.sqrt(var + eps)
    xhat = (xd - mu) * rstd
    out = xhat * gamma.data + beta.data

    def bw(g):
        lead = tuple(range(g.ndim - 1))
        dgamma = (g * xhat).sum(axis=lead)
        dbeta = g.sum(axis=lead)
        dxhat = g * gamma.data
        dx = rstd * (dxhat - dxhat.mean(axis=-1, keepdims=True)
                     - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
        return dx, dgamma, dbeta
    return Tensor._make(out, (x, gamma, beta), bw)


def conv2d(x, w, b, stride=1, pad=1):
    """Channels-last convolution. x (N,H,W,C), w (kh,kw,C,O), b (O,) -> (N,oh,ow,O).
    im2col layout is chosen so no transposes/copies are needed in forward or backward."""
    N, H, W, C = x.shape
    kh, kw, _, O = w.shape
    s = stride
    xpad = xp.pad(x.data, ((0, 0), (pad, pad), (pad, pad), (0, 0))) if pad else x.data
    oh = (H + 2 * pad - kh) // s + 1
    ow = (W + 2 * pad - kw) // s + 1
    cols = xp.empty((N, oh, ow, kh, kw, C), dtype=DTYPE)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, :, i, j, :] = xpad[:, i:i + s * oh:s, j:j + s * ow:s, :]
    cols2 = cols.reshape(N * oh * ow, kh * kw * C)
    wmat = w.data.reshape(kh * kw * C, O)
    out = (cols2 @ wmat + b.data).reshape(N, oh, ow, O)

    def bw(g):
        g2 = g.reshape(-1, O)
        dw = (cols2.T @ g2).reshape(w.shape)
        db = g2.sum(axis=0)
        dcols = (g2 @ wmat.T).reshape(N, oh, ow, kh, kw, C)
        dxpad = xp.zeros_like(xpad)
        for i in range(kh):
            for j in range(kw):
                dxpad[:, i:i + s * oh:s, j:j + s * ow:s, :] += dcols[:, :, :, i, j, :]
        dx = dxpad[:, pad:pad + H, pad:pad + W, :] if pad else dxpad
        return dx, dw, db
    return Tensor._make(out, (x, w, b), bw)


def upsample2x(x):
    """Nearest-neighbour 2x upsample, channels-last (N,H,W,C)."""
    N, H, W, C = x.shape
    out = x.data.repeat(2, axis=1).repeat(2, axis=2)
    return Tensor._make(out, (x,), lambda g: (g.reshape(N, H, 2, W, 2, C).sum(axis=(2, 4)),))
