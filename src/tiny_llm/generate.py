import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
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

    while (token_id := output.item()) != tokenizer.eos_token_id: # type: ignore[reportAttributeAccessIssue]
        detokenizer.add_token(token=token_id) # type: ignore
        print(detokenizer.last_segment, end="", flush=True) #type: ignore
        # concatnete along colomn, or last axis
        tokens = mx.concatenate([tokens, output.reshape(1, 1)], axis=-1)

        output = _step(model, tokens)

    detokenizer.finalize() #type: ignore
    print(detokenizer.last_segment, end="", flush=True) #type: ignore

    return detokenizer.text




    # tokens = tokenizer.encode()


def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass




def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
