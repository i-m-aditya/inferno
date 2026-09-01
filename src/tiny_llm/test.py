import mlx.core as mx
# logprobs = mx.array([5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0])  # n=7
# n = logprobs.shape[-1]
# top_k = 3
# k = n - top_k + 1
# print('n =', n, 'top_k =', top_k, 'k =', k)
# indices = mx.argpartition(logprobs, k, axis=-1)
# print('indices:', indices)
# print('values at indices:', logprobs[indices])
# keep = indices[k:]
# print('keep (user code):', keep, 'values:', logprobs[keep])

# # what SHOULD top-3 be?
# sorted_desc = mx.argsort(-logprobs)
# print('true top-3 indices:', sorted_desc[:3], 'values:', logprobs[sorted_desc[:3]])


data = mx.array([1,2,3])

print("rsqrt: ", data.rsqrt())


from datetime import datetime
from tiny_llm.batch import Request, _print_progress


def make_dummy_request(prompt_idx, offset, text, is_prefill_done=True, prefill_size=10):
    req = Request.__new__(Request)
    req.prompt_idx = prompt_idx
    req.offset = offset
    req.is_prefill_done = is_prefill_done
    req.prefill_tokens = mx.zeros(prefill_size)  # only `.size` is read
    req.detokenizer = type("FakeDetokenizer", (), {"text": text})()
    return req


decode_requests = [
    make_dummy_request(0, 42, "The quick brown fox jumps over the lazy dog. " * 3),
    None,
    make_dummy_request(2, 7, "Once upon a time"),
    None,
    make_dummy_request(4, 100, "def foo():\n    return 1"),
]

pending = make_dummy_request(5, 3, "", is_prefill_done=False, prefill_size=20)

print("\n--- _print_progress test ---")
_print_progress(
    requests=decode_requests,
    pending_prefill_request=pending,
    queue_size=7,
    progress_cnt=2,
    start_time=datetime.now(),
)

print("\n--- with no pending prefill ---")
_print_progress(
    requests=decode_requests,
    pending_prefill_request=None,
    queue_size=0,
    progress_cnt=3,
    start_time=datetime.now(),
)
