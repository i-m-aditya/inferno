import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from tiny_llm.kv_cache import TinyKvFullCache
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        # 1 x S x vocab_sze
        output_logits = model(y)

        # 1 x vocab
        # -1 means for last token, give me the vocab
        # # rule of thumb, when we use such representation
        # -1 or 1 or whatever number, remove that dimension, and
        # give the output for  that fixed dimension
        logits = output_logits[:,-1,:]

        if sampler is not None:
            # for every batch, index of the highest value
            indices = sampler(logits)
        else:
            indices = mx.argmax(logits, axis=-1)

        return indices

    # this will work at run time
    # it is the known limitation of __getattr__ based forwarding
    #
    # 1 x S
    tokens = mx.array([tokenizer.encode(prompt)])  # type: ignore[reportAttributeAccessIssue]

    # 1 x 1
    output = _step(model, tokens)

    detokenizer = tokenizer.detokenizer
    detokenizer.reset() #type: ignore

    while (token_id := output.item()) != tokenizer.eos_token_id: # typype: ignore[reportAttributeAccessIssue]
        detokenizer.add_token(token=token_id) # type: ignore
        print(detokenizer.last_segment, end="", flush=True) #type: ignore
        # concatnete along colomn, or last axis
        tokens = mx.concatenate([tokens, output.reshape(1, 1)], axis=-1)

        output = _step(model, tokens)

    detokenizer.finalize() #type: ignore
    print(detokenizer.last_segment, end="", flush=True) #type: ignore

    return detokenizer.text #type: ignore


def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        # 1 x S x vocab_sze
        output_logits = model(y, offset, cache=kv_cache)

        # 1 x vocab
        # -1 means for last token, give me the vocab
        # # rule of thumb, when we use such representation
        # -1 or 1 or whatever number, remove that dimension, and
        # give the output for  that fixed dimension
        logits = output_logits[:,-1,:]

        indices = mx.argmax(logits, axis=-1)
        return indices

    # this will work at run time
    # it is the known limitation of __getattr__ based forwarding
    #
    # 1 x S
    tokens = mx.array([tokenizer.encode(prompt)])  # type: ignore[reportAttributeAccessIssue]

    kv_cache = []
    for i in range(model.num_hidden_layers):
        kv_cache.append(TinyKvFullCache())

    # prefill: full prompt at offset 0
    output = _step(model, tokens, 0, kv_cache)
    offset = tokens.shape[-1]

    detokenizer = tokenizer.detokenizer
    detokenizer.reset() #type: ignore

    while (token_id := output.item()) != tokenizer.eos_token_id: # type: ignore[reportAttributeAccessIssue]
        detokenizer.add_token(token=token_id) # type: ignore
        print(detokenizer.last_segment, end="", flush=True) #type: ignore

        # decode: feed only the newly generated token, cache holds the rest
        next_token = output.reshape(1, 1)
        output = _step(model, next_token, offset, kv_cache)
        offset += 1

    detokenizer.finalize() #type: ignore
    print(detokenizer.last_segment, end="", flush=True) #type: ignore

    return detokenizer.text #type: ignore




def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
