import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array, # N.. x I
    w: mx.array, # O x I
    bias: mx.array | None = None,
) -> mx.array:

    res = None
    if bias is not None:
        res = mx.matmul(x, mx.swapaxes(w, -1, -2)) + bias
    else:
        res = mx.matmul(x, mx.swapaxes(w, -1, -2))
    return res


def silu(x: mx.array) -> mx.array:

    sigmoid = 1 / (1 + mx.exp(-mx.abs(x)))

    sigmoid_neg = 1 - sigmoid

    condition = x > 0

    silu = mx.where(condition, sigmoid, sigmoid_neg)

    return x * silu
