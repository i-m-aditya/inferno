import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper, StreamingDetokenizer
from .kv_cache import BatchingKvCache, TinyKvFullCache
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Any, Callable, cast
from datetime import datetime


def _step(model, y, offsets, kv_cache):
    logits = model(y, offsets, kv_cache)
    logits = logits[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    sampler = lambda x: mx.argmax(x, axis=-1)
    y = sampler(logprobs)
    return y


class Request:
    def __init__(
        self,
        model: Any,
        tokenizer: TokenizerWrapper,
        prompt: str,
        prefill_max_step: int = 128,
        prompt_idx: int = 0,
    ):
        self.prompt = prompt
        self.kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
        self.model = model
        self.detokenizer: StreamingDetokenizer = cast(
            StreamingDetokenizer, tokenizer.detokenizer.__class__(tokenizer._tokenizer)
        )
        self.prefill_tokens = mx.array(
            tokenizer.encode(prompt, add_special_tokens=False)
        )
        self.prefill_max_step = prefill_max_step
        self.is_done = False
        self.is_prefill_done = False
        self.eos_token_id = tokenizer.eos_token_id
        self.next_token = None
        self.offset = 0
        self.prompt_idx = prompt_idx

    def try_prefill(self):
        """
        Prefill this request up to max_step size, returns None if prefill is not done
        """
        if self.is_prefill_done:
            raise ValueError("prefill called after done")
        # Task 4: prefill the full request at once; Task 5 will chunk this.

        chunk = self.prefill_tokens[
            self.offset : self.offset + self.prefill_max_step
        ].reshape((1, -1))

        token = _step(self.model, chunk, self.offset, self.kv_cache)
        self.next_token = token.item()

        self.offset += chunk.size
        if self.offset >= self.prefill_tokens.size:
            self.is_prefill_done = True

        # to fix lazy graph from growing without bounds
        mx.eval([c.key_values for c in self.kv_cache])

    def decode_done(self, token, update_offset=True):
        if self.is_done:
            raise ValueError("decode called after done")
        if token == self.eos_token_id:
            self.is_done = True
            return
        # TODO: update the offset and add the token to the detokenizer
        if update_offset:
            self.offset += 1
        self.detokenizer.add_token(token=token)

    def text(self):
        return self.detokenizer.text


def _print_progress(
    requests: list[Request | None],
    pending_prefill_request: Request | None,
    queue_size: int,
    progress_cnt: int,
    start_time: datetime,
):
    print(f"  --- {datetime.now() - start_time}")
    animation_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    animation_frame = animation_frames[progress_cnt % len(animation_frames)]
    for i, request in enumerate(requests):
        if request is None:
            print(f"  Decode #{i}: idle", flush=True)
        else:
            text_preview = request.text()[-80:].replace("\n", " ")
            print(
                f"{animation_frame} Decode [req {request.prompt_idx}, {request.offset}]: {text_preview}",
                flush=True,
            )
    if pending_prefill_request is not None:
        if pending_prefill_request.is_prefill_done:
            print(
                f"  Prefill [req {pending_prefill_request.prompt_idx}]: done, waiting for slot, {queue_size} requests in queue",
                flush=True,
            )
            return
        precentage = (
            pending_prefill_request.offset / pending_prefill_request.prefill_tokens.size
        ) * 100
        print(
            f"{animation_frame} Prefill [req {pending_prefill_request.prompt_idx}]: {precentage:.2f}% ({pending_prefill_request.prefill_tokens.size - pending_prefill_request.offset} remaining tokens)",
            flush=True,
        )
    else:
        print(f"  Prefill: idle, {queue_size} requests in queue", flush=True)


def batch_generate(
    model: Any,
    tokenizer: TokenizerWrapper,
    prompts: list[str],
    max_seq_len=512,
    batch_size=5,
    prefill_step=128,
):
    decode_requests: list[Request | None] = [None] * batch_size
    kv_cache = [
        BatchingKvCache(max_active_requests=batch_size, max_seq_len=max_seq_len)
        for _ in range(model.num_hidden_layers)
    ]
    result = []
    pending_prefill_request = None
    next_request_idx = 0
    progress_cnt = 0
    start_time = datetime.now()

    while True:
        if len(prompts) == 0 and all(req is None for req in decode_requests):
            break
        # prefill until no idle slots
        if len(prompts) > 0 and pending_prefill_request is None:
            prompt = prompts.pop(0)
            pending_prefill_request = Request(
                model, tokenizer, prompt, prefill_step, next_request_idx
            )
            next_request_idx += 1

        # In every iteration, we do a prefill first
        if pending_prefill_request is not None:
            made_progress = False
            if not pending_prefill_request.is_prefill_done:
                pending_prefill_request.try_prefill()
                made_progress = True
            if pending_prefill_request.is_prefill_done:
                idle_slot = None
                for req_idx in range(len(decode_requests)):
                    if decode_requests[req_idx] is None:
                        idle_slot = req_idx
                        break

                if idle_slot is not None:
                    decode_requests[idle_slot] = pending_prefill_request
                    for layer in range(len(kv_cache)):
                        kv_cache[layer].add_request(
                            pending_prefill_request.kv_cache[layer], idle_slot
                        )
                    pending_prefill_request = None
                # else: no free slot yet -- leave pending_prefill_request as-is
                # (waiting), and let the decode step below still run normally
                # for whichever requests are already active.

            if made_progress:
                _print_progress(
                    decode_requests,
                    pending_prefill_request,
                    len(prompts),
                    progress_cnt,
                    start_time,
                )
                progress_cnt += 1

        # After the prefill request moves forward one step, we do the decode
        if any(req is not None for req in decode_requests):
            next_tokens = []
            offsets = []
            # collect the next tokens and offsets from the decode requests
            for _slot, request in enumerate(decode_requests):
                if request is not None:
                    next_tokens.append(request.next_token)
                    offsets.append(request.offset)
                    continue
                next_tokens.append(0)
                offsets.append(0)

            next_tokens = _step(
                model, mx.array(next_tokens).reshape(-1, 1), offsets, kv_cache
            )
            for req_idx in range(batch_size):
                # TODO: check if the decode has finished by comparing EOS or the seqlength. If so,
                # remove the request from the decode requests and add the result to the result list;
                # otherwise, call `decode_done` to update the offset and add the token to the detokenizer

                token = next_tokens[req_idx].item()  # extract the scalar
                req = decode_requests[req_idx]
                if req is None:
                    continue

                req.decode_done(token=token)
                if req.is_done or req.offset >= max_seq_len:
                    decode_requests[req_idx] = None
                    result.append((req.prompt_idx, req.text()))

                    for req_idx in range(len(kv_cache)):
                        kv_cache[req_idx].remove_request(req_idx)

                else:
                    req.next_token = token
                    decode_requests[req_idx] = req

            _print_progress(
                decode_requests,
                pending_prefill_request,
                len(prompts),
                progress_cnt,
                start_time,
            )
            progress_cnt += 1
    return result
