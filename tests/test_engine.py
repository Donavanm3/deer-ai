"""Numerical gradient checks for every hand-written op in the engine."""
import numpy as np
import pytest
from deer.engine import Tensor, concat, embedding, cross_entropy, soft_cross_entropy, bce_with_logits_sum, layer_norm, conv2d, upsample2x
from deer.engine.tensor import DTYPE

rng = np.random.default_rng(0)


def numgrad(f, arrs, i, eps=1e-3):
    a = arrs[i]
    g = np.zeros_like(a, dtype=np.float64)
    it = np.nditer(a, flags=["multi_index"])
    for _ in it:
        ix = it.multi_index
        old = a[ix]
        a[ix] = old + eps; fp = f(*arrs)
        a[ix] = old - eps; fm = f(*arrs)
        a[ix] = old
        g[ix] = (fp - fm) / (2 * eps)
    return g


def check(fn, *shapes, tol=2e-2):
    arrs = [rng.normal(0, 1, s).astype(np.float64) for s in shapes]
    w = None

    def scalar(*xs):
        nonlocal w
        ts = [Tensor(np.asarray(x, dtype=DTYPE)) for x in xs]
        out = fn(*ts)
        if w is None:
            w = rng.normal(0, 1, out.shape).astype(DTYPE)
        return float((out.data.astype(np.float64) * w).sum())
    scalar(*arrs)
    ts = [Tensor(a.astype(DTYPE), requires_grad=True) for a in arrs]
    out = fn(*ts)
    (out * Tensor(w)).sum().backward()
    for i, t in enumerate(ts):
        ng = numgrad(scalar, arrs, i)
        ag = t.grad.astype(np.float64)
        err = np.abs(ng - ag).max() / (np.abs(ng).max() + 1e-6)
        assert err < tol, f"input {i}: rel err {err}"


def test_arith():
    check(lambda a, b: a * b + a - b / (b * b + 2), (3, 4), (1, 4))
    check(lambda a: a ** 3, (5,))


def test_matmul_batched():
    check(lambda a, b: a @ b, (2, 3, 4), (4, 5))
    check(lambda a, b: a @ b.transpose(0, 2, 1), (2, 3, 4), (2, 5, 4))


def test_reductions_shapes():
    check(lambda a: a.sum(axis=1), (3, 4))
    check(lambda a: a.mean(axis=(2, 3)), (2, 3, 2, 2))
    check(lambda a: a.reshape(4, 3).transpose(1, 0), (3, 4))
    check(lambda a: a[1], (3, 4))


def test_activations():
    check(lambda a: a.relu() + a.tanh() + a.sigmoid() + a.gelu() + a.exp(), (4, 5))
    check(lambda a: a.softmax(-1), (3, 6))
    check(lambda a: (a * a + 1).log(), (3,))


def test_masked_fill_concat():
    mask = np.triu(np.ones((4, 4), bool), 1)
    check(lambda a: a.masked_fill(mask, -1e9).softmax(-1), (2, 4, 4))
    check(lambda a, b: concat([a, b], axis=1), (2, 3), (2, 2))


def test_losses():
    tgt = np.array([1, 0, 3])
    check(lambda a: cross_entropy(a, tgt), (3, 5))
    t = rng.random((2, 6)).astype(DTYPE)
    check(lambda a: bce_with_logits_sum(a, t, 2), (2, 6))
    soft = rng.random((3, 5)).astype(DTYPE); soft /= soft.sum(1, keepdims=True)
    check(lambda a: soft_cross_entropy(a, soft), (3, 5))


def test_layer_norm():
    check(lambda x, g, b: layer_norm(x, g, b), (2, 3, 6), (6,), (6,))


def test_conv_upsample():
    check(lambda x, w, b: conv2d(x, w, b, 1, 1), (2, 5, 5, 2), (3, 3, 2, 3), (3,))
    check(lambda x, w, b: conv2d(x, w, b, 2, 1), (1, 6, 6, 2), (3, 3, 2, 2), (2,))
    check(lambda x: upsample2x(x), (1, 3, 3, 2))


def test_embedding():
    W = Tensor(rng.normal(0, 1, (5, 3)).astype(DTYPE), requires_grad=True)
    idx = np.array([[0, 2, 2], [4, 0, 1]])
    embedding(W, idx).sum().backward()
    expected = np.zeros((5, 3)); np.add.at(expected, idx.reshape(-1), 1)
    assert np.allclose(W.grad, expected)


def test_shared_node_grad_accumulates():
    a = Tensor(np.array([2.0], dtype=DTYPE), requires_grad=True)
    b = a * a
    (b + b).backward()
    assert np.allclose(a.grad, [8.0])


def test_kv_cache_matches_full_forward():
    """Fast cached generation must pick exactly the same characters as the slow full recompute."""
    from deer.models import DeerChat, build_config
    from deer.engine import no_grad
    m = DeerChat(build_config("chat", "tiny", {"block_size": 64}))
    prompt = list(b"User: Hello there, Deer!\nDeer:")
    ids, out = list(prompt), []
    with no_grad():
        for _ in range(30):  # greedy, recomputing the whole context every step
            nxt = int(np.argmax(m.forward(np.array([ids])).data[0, -1]))
            ids.append(nxt); out.append(nxt)
    assert m.generate(bytes(prompt), max_new=30, top_k=1) == bytes(out).decode("utf-8", "replace")
    # past the context window it keeps going (window slides) and stops on the stop byte
    assert len(m.generate(bytes(prompt) * 3, max_new=100, top_k=5, seed=1)) > 0
