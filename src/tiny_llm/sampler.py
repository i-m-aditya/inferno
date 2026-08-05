import mlx.core as mx

def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)

        if top_k is not None:
            n = logprobs.shape[-1]
            k = n - top_k

            indices = mx.argpartition(logprobs, k, axis=-1)
            keep = indices[..., k:]

            mask = mx.zeros(logprobs.shape, dtype=mx.bool_)
            mask = mx.put_along_axis(
                mask, keep, mx.ones(keep.shape, dtype=mx.bool_), axis=-1
            )

            logprobs = mx.where(mask, logprobs, -mx.inf)

        if top_p is not None:
            probs = mx.exp(logprobs)
            # sort in ascending order
            sorted_indices = mx.argsort(logprobs, axis=-1)
            sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

            cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

            # rearrange cumulative probs back to original order
            inverse_indices = mx.put_along_axis(
                mx.zeros_like(sorted_indices),
                sorted_indices,
                mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
                axis=-1,
            )
            cumulative_probs = mx.take_along_axis(
                cumulative_probs, inverse_indices, axis=-1
            )

            # keep tokens whose cumulative prob (from the smallest up) exceeds 1 - top_p,
            # i.e. the smallest set of highest-probability tokens covering top_p mass
            logprobs = mx.where(
                cumulative_probs > 1 - top_p,
                logprobs,
                -mx.inf,
            )

        return mx.random.categorical(logprobs / temp, axis=-1)

    return sample
