import math

import mlx.core as mx
from typing_extensions import final
from .basics import linear, silu
from .attention import flash_attention, scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights, quantized_linear
from .kv_cache import TinyKvCache

@final
class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_flash_attention: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = head_dim

        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

        self.q_norm = q_norm
        self.k_norm = k_norm

        self.rope = RoPE(
            dims=head_dim,
            seq_len=max_seq_len,
            base=theta,
            traditional=False
        )

        self.rms_norm_eps = rms_norm_eps
        self.use_flash_attention = use_flash_attention

    def __call__(
        self,
        x: mx.array,
        offset: int | list[int],
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, E = x.shape
        # B, L, H_q, D
        q = quantized_linear(x, self.wq).reshape(B, L, self.num_heads, self.head_dim)

        # B, L, H, D
        k = quantized_linear(x, self.wk).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = quantized_linear(x, self.wv).reshape(B, L, self.num_kv_heads, self.head_dim)


        q = mx.fast.rms_norm(
            q, self.q_norm, self.rms_norm_eps
        )

        k = mx.fast.rms_norm(
            k, self.k_norm, self.rms_norm_eps
        )


        if isinstance(offset, int):

            # B, H_q, L, D
            q = self.rope(
                x=q, offset=slice(offset, offset + L)
            ).swapaxes(-3,-2).astype(mx.float32)

            # B, H, L, D
            k = self.rope(
                x=k, offset=slice(offset, offset + L)
            ).swapaxes(-3, -2).astype(mx.float32)
        else:
            q = self.rope(
                x=q,
                offset=[slice(off, off+L) for off in offset]
            ).swapaxes(-3,-2).astype(mx.float32)

            # B, H, L, D
            k = self.rope(
                x=k,
                offset=[slice(off, off+L) for off in offset]
            ).swapaxes(-3, -2).astype(mx.float32)

        # B, H, L, D
        v = v.swapaxes(-3, -2).astype(mx.float32)


        # caching (k/v are already rope'd; cache accumulates rotated K, raw V)
        k, v, cache_offset, cache_mask = cache.update_and_fetch(k, v, L, None)
        if cache_offset is not None:
            assert cache_offset - L == offset, "offset passed in must match cache's prior length"

        # BatchingKvCache builds its own padding-aware mask; TinyKvFullCache
        # never does (always returns None here), so fall back to the mask
        # passed in from the model level in that case.
        effective_mask = cache_mask if cache_mask is not None else mask

        if self.use_flash_attention:
            attention = flash_attention(
                query=q,
                key=k,
                value=v,
                scale=1/math.sqrt(self.head_dim),
                mask=effective_mask
            ).astype(x.dtype)
        else:
            attention = scaled_dot_product_attention_grouped(
                query=q,
                key=k,
                value=v,
                scale=1/math.sqrt(self.head_dim),
                mask=effective_mask
            ).astype(x.dtype)

        # B L Hq D
        output = attention.swapaxes(-3, -2)

        # B L (Hq * D)
        output = output.reshape(B, L, (self.num_heads*self.head_dim))

        output = quantized_linear(output, self.wo)

        return output


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_down = w_down
        self.w_up = w_up

    def __call__(self, x: mx.array) -> mx.array:


       u = quantized_linear(x, self.w_up)

       g  = silu(quantized_linear(x, self.w_gate))

       out = quantized_linear(g * u, self.w_down)

       return out

class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):

        self.attention = Qwen3MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            max_seq_len,
            theta,
            rms_norm_eps,
            use_flash_attention
        )

        self.mlp = Qwen3MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down
        )
        self.input_layernorm = RMSNorm(
            dim=hidden_size,
            weight=w_input_layernorm,
            eps=rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            dim=hidden_size,
            weight=w_post_attention_layernorm,
            eps=rms_norm_eps
        )

        self.use_flash_attention = use_flash_attention

    def __call__(
        self,
        x: mx.array,
        offset: int | list[int],
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # normalized
        normalized_x = self.input_layernorm(x)
        attention = self.attention(normalized_x, offset, cache=cache, mask=mask)
        x = x + attention

        normalized_x = self.post_attention_layernorm(x)
        mlp = self.mlp(normalized_x)

        x = x + mlp

        return x

class Qwen3ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        enable_flash_attn: bool = False,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        precision = mx.bfloat16
        self.precision = precision

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens),
        )
        self.layers = []
        for i in range(mlx_model.args.num_hidden_layers):
            layer = Qwen3TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                head_dim=mlx_model.args.head_dim,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=mlx_model.model.layers[i].self_attn.q_proj,
                wk=mlx_model.model.layers[i].self_attn.k_proj,
                wv=mlx_model.model.layers[i].self_attn.v_proj,
                wo=mlx_model.model.layers[i].self_attn.o_proj,
                q_norm=mlx_model.model.layers[i].self_attn.q_norm.weight,
                k_norm=mlx_model.model.layers[i].self_attn.k_norm.weight,
                w_gate=mlx_model.model.layers[i].mlp.gate_proj,
                w_down=mlx_model.model.layers[i].mlp.down_proj,
                w_up=mlx_model.model.layers[i].mlp.up_proj,
                w_input_layernorm=mlx_model.model.layers[i].input_layernorm.weight,
                w_post_attention_layernorm=mlx_model.model.layers[i].post_attention_layernorm.weight,
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
                use_flash_attention=enable_flash_attn
            )

            self.layers.append(layer)
        self.norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight,
            eps=mlx_model.args.rms_norm_eps
        )

        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)

        self.mlx_model = mlx_model


    def __call__(
        self,
        inputs: mx.array,
        offset: int | list[int],
        cache: list[TinyKvCache],
    ) -> mx.array:
        # N.. x L x E
        x = self.embedding(inputs)
        *N, L, E = x.shape
        mask = "causal" if L > 1 else None

        for i in range(len(self.layers)):
            x = self.layers[i](x, offset, cache[i], mask)

        x = self.norm(x)


        if hasattr(self, 'w_lm_head'):
            output = linear(x, self.w_lm_head)
        else:
            output = self.embedding.as_linear(x)
        return output
