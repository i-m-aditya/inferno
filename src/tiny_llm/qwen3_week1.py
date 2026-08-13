
import mlx.core as mx
from sympy.geometry.util import deque
from tiny_llm.embedding import Embedding
from tiny_llm.layer_norm import RMSNorm
from .quantize import dequantize_linear
from .basics import linear, silu
from .attention import causal_mask, scaled_dot_product_attention_grouped
from .positional_encoding import RoPE
from typing import Any
import math


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim


        self.num_kv_heads = num_kv_heads
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

        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:

        B, L, E = x.shape
        # B, L, H_q, D
        q = linear(x, self.wq).reshape(B, L, self.num_heads, self.head_dim)

        # B, L, H, D
        k = linear(x, self.wk).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = linear(x, self.wv).reshape(B, L, self.num_kv_heads, self.head_dim)


        q = mx.fast.rms_norm(
            q, self.q_norm, self.rms_norm_eps
        )

        k = mx.fast.rms_norm(
            k, self.k_norm, self.rms_norm_eps
        )

        '''
            Apply rope to q and k, v never gets rope
        '''
        # B, H_q, L, D
        q = self.rope(
            x=q, offset=slice(0, L)
        ).swapaxes(-3,-2).astype(mx.float32)

         # B, H, L, D
        k = self.rope(
            x=k, offset=slice(0, L)
        ).swapaxes(-3, -2).astype(mx.float32)

        # B, H, L, D
        v = v.swapaxes(-3, -2).astype(mx.float32)

        attention = scaled_dot_product_attention_grouped(
            query=q,
            key=k,
            value=v,
            scale=1/math.sqrt(self.head_dim),
            mask=mask
        ).astype(x.dtype)

        # B L Hq D
        output = attention.swapaxes(-3, -2)

        # B L (Hq * D)
        output = output.reshape(B, L, (self.num_heads*self.head_dim))

        output = linear(output, self.wo)

        return output

class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_down = w_down
        self.w_up = w_up

    def __call__(self, x: mx.array) -> mx.array:

        u = linear(x, self.w_up)

        g  = silu(linear(x, self.w_gate))

        out = linear(g * u, self.w_down)

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
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
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
            theta
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

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # normalized
        normalized_x = self.input_layernorm(x)
        attention = self.attention(normalized_x, mask)
        x = x + attention

        normalized_x = self.post_attention_layernorm(x)
        mlp = self.mlp(normalized_x)

        x = x + mlp

        return x


class Qwen3ModelWeek1:
    # `mlx_model` we currently rely on this to load weights to memory
    def __init__(self, mlx_model: Any):
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
                wq=dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj),
                wk=dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj),
                wv=dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj),
                wo=dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj),
                q_norm=mlx_model.model.layers[i].self_attn.q_norm.weight,
                k_norm=mlx_model.model.layers[i].self_attn.k_norm.weight,
                w_gate=dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj),
                w_up=dequantize_linear(mlx_model.model.layers[i].mlp.up_proj),
                w_down=dequantize_linear(mlx_model.model.layers[i].mlp.down_proj),
                w_input_layernorm=mlx_model.model.layers[i].input_layernorm.weight,
                w_post_attention_layernorm=mlx_model.model.layers[
                    i
                ].post_attention_layernorm.weight,
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
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
    ) -> mx.array:
        # N.. x L x E
        h = self.embedding(inputs)
        *N, L, E = h.shape
        mask = "causal" if L > 1 else None

        for layer in self.layers:
            h = layer(h, mask)

        h = self.norm(h)


        if hasattr(self, 'w_lm_head'):
            output = linear(h, self.w_lm_head)
        else:
            output = self.embedding.as_linear(h)
        return output
