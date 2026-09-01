from abc import ABC, abstractmethod
from typing import Optional

import mlx.core as mx
from typing_extensions import final

from .attention import causal_mask


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int | None, Optional[mx.array]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            A tuple of the updated key-value cache, the updated value, the sequence length, and the mask.
            In week 2 day 1, we only need to return the updated key-value cache, the updated value.
            In week 2 day 6/7, we need to return the updated key-value cache, the updated value, the sequence length, and the mask.
            so that the batching kv cache can use this information to generate the mask.
        """


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len
        # each active slot owns its own cache (same kind TinyKvFullCache uses
        # for a single request) -- we delegate to it rather than duplicating
        # its accumulate-history logic here.
        self.slots: list[TinyKvCache | None] = [None] * max_active_requests

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int | None, Optional[mx.array]]:
        assert mask_length is not None  # BatchingKvCache always needs a real mask length
        active_ids = [i for i in range(self.max_active_requests) if self.slots[i] is not None]

        # Step (a): delegate to each active slot's own cache to append this
        # step's new token and grow its history -- reuses TinyKvFullCache's
        # own update_and_fetch instead of re-implementing concatenation here.
        updated: dict[int, tuple[mx.array, mx.array, int]] = {}
        for i in active_ids:
            slot = self.slots[i]
            assert slot is not None  # active_ids already guarantees this
            key_i, value_i, S_i, _ = slot.update_and_fetch(keys[i : i + 1], values[i : i + 1])
            assert S_i is not None
            updated[i] = (key_i, value_i, S_i)

        S = max(S_i for _, _, S_i in updated.values())

        _, H, _, D = keys.shape
        batched_keys = mx.zeros((self.max_active_requests, H, S, D), dtype=keys.dtype)
        batched_values = mx.zeros((self.max_active_requests, H, S, D), dtype=values.dtype)
        out_mask = mx.full(
            (self.max_active_requests, 1, mask_length, S), float("-inf"), dtype=keys.dtype
        )

        for i in active_ids:
            key_i, value_i, S_i = updated[i]
            # use i:i+1 (not scalar i) -- keeps the leading dim, matching key_i's
            # own shape (1, H, S_i, D); MLX's indexed assignment doesn't like a
            # scalar int index combined with a value that still has that axis.
            batched_keys[i : i + 1, :, (S - S_i) : S, :] = key_i # assigning both are of same shape
            batched_values[i : i + 1, :, (S - S_i) : S, :] = value_i
            out_mask[i : i + 1, :, :, (S - S_i) : S] = causal_mask(
                mask_length, S_i, dtype=keys.dtype
            )

        return (batched_keys, batched_values, None, out_mask)

    def add_request(self, prefilled: TinyKvCache, id: int):
        self.slots[id] = prefilled

    def remove_request(self, id: int):
        self.slots[id] = None

@final
class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key_values: tuple[mx.array, mx.array] | None = None
        self.offset = 0


    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        #key:   B, H, S_new, D
        #value: B, H, S_new, D

        if self.key_values is None:
            self.key_values = (key, value)
        else:
            cached_keys, cached_values = self.key_values
            cached_keys = mx.concat([cached_keys, key], axis=2)

            cached_values = mx.concat([cached_values, value], axis=2)

            self.key_values = (cached_keys, cached_values)

        self.offset += key.shape[2]

        key, value = self.key_values

        return (key, value, self.offset, None)
