import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0, "dims must be even"
        # Step 1: frequency per pair index i, i = 0 .. D//2 - 1
        # theta_i = base ** (-2i / D)
        inner = mx.arange(0, dims, 2) / dims

        freqs = base ** (-inner)

        # step 2: positions 0 .. seq_len-1
        pos = mx.arange(seq_len) # shape(seq_len)

        # step 3: outer product -> angles[pos, i] = pos * freq[i]
        angles = mx.outer(pos, freqs)

        self.cos_freqs = mx.cos(angles) # L x D/2
        self.sin_freqs = mx.sin(angles) # L x D/2
        self.traditional = traditional

    '''
    Overall desired outcome of `__call__`

    Input: `x` — a raw query or key tensor, shape `N x L x H x D`,
    with no positional information encoded (attention alone is position-blind, as discussed).
    Plus `offset`, telling you which absolute positions this `x`'s `L` tokens correspond to.

    Output: same shape `N x L x H x D`, but now each token's `D`-dim vector
    has been rotated — each of its `D/2` pairs spun by an angle proportional to that token's position
    and that pair's frequency. This rotated tensor is what gets passed into
    `scaled_dot_product_attention_simple` in place of the raw `q`/`k` — so that when `QK^T` computes dot products,
    those scores end up encoding *relative* position between query and key tokens,
    per the math we discussed earlier.
    '''
    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:

        N, L, H, D = x.shape

        if offset is None:
            cos = self.cos_freqs[:L] #
            sin = self.sin_freqs[:L]
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        elif isinstance(offset, list):
            cos = []
            sin = []
            for off in offset:
                # slice bounds may come in as np.int64 (e.g. from iterating a
                # numpy array) -- MLX indexing requires plain Python ints.
                off = slice(
                    int(off.start) if off.start is not None else None,
                    int(off.stop) if off.stop is not None else None,
                    int(off.step) if off.step is not None else None,
                )
                cos.append(self.cos_freqs[off])
                sin.append(self.sin_freqs[off])

            cos = mx.stack(cos, axis=0)
            sin = mx.stack(sin, axis=0)

            cos = cos[:, :, None, :]
            sin = sin[:, :, None, :]
        else:
            cos = self.cos_freqs[offset]
            sin = self.sin_freqs[offset]
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]

        if self.traditional:
            x1 = x[..., 0::2] # even indices
            x2 = x[..., 1::2] # odd indices
        else:
            x1 = x[..., :D // 2]
            x2 = x[..., D // 2 :]

        '''
        The naming comes from an equivalent way of thinking about RoPE
        that the papers use: a 2D pair `(x1, x2)` can be
        viewed as a single **complex number** `x1 + i·x2`,
        where `x1` is the "real part" and `x2` is the "imaginary part."

        (x1 + i·x2) * (cos θ + i·sin θ) = (x1·cos θ - x2·sin θ) + i·(x1·sin θ + x2·cos θ)
        '''
        real = x1 * cos - x2 * sin
        imag = x1 * sin + x2 * cos

        if self.traditional:
            out = mx.stack([real, imag], axis=-1).reshape(N, L, H, D)

        else:
            out = mx.concatenate([real, imag], axis=-1)

        return out.astype(x.dtype)
