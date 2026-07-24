import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):

        self.dim = dim
        self.weight = weight
        self.eps = eps
        pass

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(mx.square(x.astype(mx.float32)), keepdims=True, axis=-1) + self.eps)
        output = (x / rms.astype(x.dtype)) * self.weight
        return output
