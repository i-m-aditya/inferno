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
