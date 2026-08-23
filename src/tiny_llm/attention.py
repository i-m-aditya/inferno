import math

import mlx.core as mx
from .basics import softmax, linear
from extensions import tiny_llm_ext

def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:

    # query: N * H * L * D
    # key: N * H * S * D
    key_t = mx.swapaxes(key, -1, -2)

    q_kt = mx.matmul(query, key_t)  # dim: N x H x L x S

    if scale is None:
        d = query.shape[-1]
        scale = 1 / math.sqrt(d)

    scores = scale * q_kt
    if mask is not None:
        scores = scores + mask  #N x H x L x S

    intermediate = softmax(scores, axis=-1)
    attention = mx.matmul(intermediate, value)

    return attention

class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array, # (H x D) x E
        wk: mx.array, # (H x D) x E
        wv: mx.array, # (H x D) x E
        wo: mx.array, # E x (H x D)
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.head_dim = hidden_size // num_heads

    def __call__(
        self,
        query: mx.array, #N x L x E
        key: mx.array, #N x L x E
        value: mx.array, #N x L x E
        mask: mx.array | None = None,
    ) -> mx.array:

        N, L, _ = query.shape
        assert query.shape == key.shape == value.shape
        d = self.hidden_size // self.num_heads

        # N H L D
        projection_q = linear(
            query, self.wq
        ).reshape(N, L, self.num_heads, self.head_dim).swapaxes(-3, -2)

        projection_k = linear(
            key, self.wk
        ).reshape(N, L, self.num_heads, self.head_dim).swapaxes(-3, -2)

        projection_v = linear(
            value, self.wv
        ).reshape(N, L, self.num_heads, self.head_dim).swapaxes(-3, -2)

        attention = scaled_dot_product_attention_simple(projection_q, projection_k, projection_v, mask=mask)  # N x H x L x D

        # N x L x H x D
        attention = mx.swapaxes(attention, -2, -3)

        # N x L x (H*D)
        attention = attention.reshape(N, L, self.num_heads * d)

        return linear(attention, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:

    query_pos = mx.arange(L)[:, None]
    key_pos = mx.arange(S)[None, :]

    allowed = key_pos <= query_pos + (S-L)

    mask = mx.where(allowed, 0.0, -mx.inf)

    mask = mask.astype(dtype)

    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    '''
    query: N.. x H_q x L x D
    key: N.. x H x S x D
    value: N.. x H x S x D
    mask: N.. x H_q x L x S
    output: N.. x H_q x L x D
    '''

    *N, h_q, L, D = query.shape

    h = key.shape[-3]
    S = key.shape[-2]

    n_repeats = h_q // h

    query = query.reshape(*N, h, n_repeats, L, D) # N h n_repeats L D

    key = key.reshape(*N, h, 1, S, D) # N h 1 S D
    value = value.reshape(*N, h, 1, S, D)


    if scale is None:
        scale = 1 / math.sqrt(D)

    if isinstance(mask, mx.array):
        # mask may arrive with a head dim of 1 (broadcast across all H_q
        # heads) -- broadcast to the full head count before reshaping,
        # since reshape can't invent new elements the way broadcast can.
        mask = mx.broadcast_to(mask, (*N, h_q, L, S))
        mask = mask.reshape(*N, h, n_repeats, L, S)

        # N h n_repeats L D
        attention = scaled_dot_product_attention_simple(
            query,
            key,
            value,
            scale,
            mask
        )
    elif mask == "causal":
        mask = causal_mask(L, S, query.dtype)
        attention = scaled_dot_product_attention_simple(
            query,
            key,
            value,
            scale,
            mask
        )
    else:
        attention = scaled_dot_product_attention_simple(
            query,
            key,
            value,
            scale,
            mask=None
        )

    attention = attention.reshape(*N, h_q, L, D)

    return attention

def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array| str | None = None,
) -> mx.array:

    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale #type: ignore
    factor  = mx.array(factor).astype(query.dtype)


    *B, H_q, L, E = query.shape

    _, H, S, _ = key.shape

    assert H_q % H == 0

    query = mx.contiguous(query.reshape(-1, L, E))
    key = mx.contiguous(key.reshape(-1, S, E))
    value = mx.contiguous(value.reshape(-1, S, E))

    N = query.shape[0]
    is_causal = mask == "causal"
    if is_causal:
        mask = causal_mask(L, S, query.dtype)
        mask = mx.broadcast_to(mask, [*B, H_q, L, S])
    elif mask is None:
        mask = mx.zeros((*B, H_q, L, S))
    else:
        mask =  mx.broadcast_to(mask, [*B, H_q, L, S])

    mask = mx.contiguous(mx.reshape(mask, (N, L, S))).astype(mx.float32)

    scores = tiny_llm_ext.flash_attention(
        query,
        key,
        value,
        mask,
        factor,
        is_causal = is_causal,
        num_heads = H_q,
        num_kv_heads = H,
    )
    scores = mx.contiguous(scores.reshape([*B, H_q, L, E]))
    return scores
