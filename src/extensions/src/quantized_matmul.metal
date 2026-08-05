// Copyright © 2023-2025 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// One thread computes exactly one output element out[i, k].
//
// a:      (M, N)       activations, row-major
// b:      (K, N/8)     packed 4-bit weights, row-major
// scales: (K, N/128)   one scale per group of 128 weights
// biases: (K, N/128)   one bias per group of 128 weights
// out:    (M, K)
template <typename T>
[[kernel]] void quantized_matmul(
    device const T* a [[buffer(0)]],
    device const uint32_t* b [[buffer(1)]],
    device const T* scales [[buffer(2)]],
    device const T* biases [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant const int& M [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& K [[buffer(7)]],
    uint2 index [[thread_position_in_grid]]) {
  // index.x walks the K (output column) axis, index.y walks the M (output row) axis --
  // this has to match how `grid_dims`/`group_dims` are set up on the C++ side in eval_gpu.
  const uint k = index.x;
  const uint i = index.y;

  // The grid may be slightly larger than (K, M) if they don't divide evenly into
  // threadgroups -- any thread landing outside the real output just does nothing.
  if (i >= (uint)M || k >= (uint)K) {
    return;
  }

  const int group_size = 128;
  const int bits = 4;
  const int packs_per_item = 32 / bits;  // 8 values packed per uint32
  const int group_per_row = N / group_size;
  const uint32_t item_mask = (1 << bits) - 1;

  float sum = 0.0;


  for (int group_idx = 0; group_idx < group_per_row; group_idx ++) {

    T scale  = scales[k*group_per_row + group_idx];
    T bias = biases[k*group_per_row + group_idx];

    // b: K x N
    // so k
    int b_start = (k * N + group_idx * group_size) / packs_per_item;

   for (int pack = 0; pack < group_size / packs_per_item; pack++) {
           uint32_t packed = b[b_start + pack];
           for (int j = 0; j < packs_per_item; j++) {
               uint32_t quantized = (packed >> (j * bits)) & item_mask;
               float dequantized = float(quantized) * float(scale) + float(bias);
               int a_col = group_idx * group_size + pack * packs_per_item + j;
               float a_val = float(a[i * (uint)N + a_col]);
               sum += a_val * dequantized;
           }
    }

  }
  out[i * (uint)K + k] = static_cast<T>(sum);
}

// clang-format off
#define instantiate_quantized_matmul(type_name, type) \
  instantiate_kernel("quantized_matmul_" #type_name, quantized_matmul, type)

instantiate_quantized_matmul(float16, half);
instantiate_quantized_matmul(bfloat16, bfloat16_t);
// clang-format on
