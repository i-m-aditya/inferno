// Copyright © 2023-2025 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

constant constexpr int FA_BR = 32;
constant constexpr int FA_BC = 32;
constant constexpr int FA_E = 128;

// One threadgroup handles one (n, i) pair -- one head/batch, one query tile.
// Inside the threadgroup: 32 SIMD groups x 32 lanes = 1024 threads.
//   simd_gid (0..31) = which query ROW within this tile
//   simd_lid (0..31) = which KEY COLUMN, within the current j tile
[[kernel]] void flash_attention_f32_e128(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& L [[buffer(6)]],
    constant int& S [[buffer(7)]],
    constant int& E [[buffer(8)]],
    constant int& num_kv_heads [[buffer(9)]],
    constant int& num_heads [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    constant bool& is_causal [[buffer(12)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int n = int(tg_pos.x);
  const int i = int(tg_pos.y);

  // Same GQA mapping as the CPU version -- every thread computes this
  // redundantly (cheap), no coordination needed.
  const int q_kv_heads_ratio = num_heads / num_kv_heads;
  const int batch_idx = n / num_heads;
  const int q_head_idx = n % num_heads;
  const int kv_head_idx = q_head_idx / q_kv_heads_ratio;
  const int n_kv = batch_idx * num_kv_heads + kv_head_idx;

  const int Tc = (S + FA_BC - 1) / FA_BC;
  const int row = i * FA_BR + int(simd_gid);  // this SIMD group's absolute query row
  const bool row_valid = row < L;

  // Threadgroup (on-chip) memory: Q for this tile, loaded once and reused
  // across every j iteration; o_i accumulates this tile's running output.
  threadgroup float q_local[FA_BR][FA_E];
  threadgroup float o_i[FA_BR][FA_E];

  // Load this row's Q vector into threadgroup memory -- the 32 lanes split
  // the E values between them, then sync so every thread in the threadgroup
  // can see the full row before it's used below. E is a runtime value here
  // (up to FA_E=128, the max this threadgroup memory can hold) -- q/k/v/out
  // are strided by the REAL E, not the compile-time FA_E.
  if (row_valid) {
    for (int e = int(simd_lid); e < E; e += FA_BC) {
      q_local[simd_gid][e] = q[(n * L + row) * E + e];
    }
  }
  for (int e = int(simd_lid); e < E; e += FA_BC) {
    o_i[simd_gid][e] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Running softmax state -- per-thread registers. simd_max/simd_sum's
  // broadcast-to-every-lane behavior keeps every thread in a row's SIMD
  // group holding the same m_i/l_i, without needing threadgroup memory.
  float m_i = -INFINITY;
  float l_i = 0.0f;

  for (int j = 0; j < Tc; j++) {
    const int col = j * FA_BC + int(simd_lid);  // this thread's key column
    const bool col_valid = col < S;

    // Causal tile classification -- every thread computes this identically
    // (it only depends on tile-level ranges, not per-thread row/col), so all
    // 1024 threads agree and take the same `continue` decision together.
    bool skip_mask = false;
    if (is_causal) {
      const int min_row = i * FA_BR;
      const int max_row = min(i * FA_BR + FA_BR - 1, L - 1);
      const int min_col = j * FA_BC;
      const int max_col = min(j * FA_BC + FA_BC - 1, S - 1);
      const int offset = S - L;
      if (min_col > max_row + offset) {
        continue;
      } else if (max_col <= min_row + offset) {
        skip_mask = true;
      }
    }

    // Step A -- S[simd_gid, simd_lid]: one thread computes one (row, col)
    // score, same "one thread per output element" mapping as
    // quantized_matmul's kernel, just reading Q from threadgroup memory
    // instead of global memory.
    float s = -INFINITY;
    if (row_valid && col_valid) {
      s = 0.0f;
      for (int e = 0; e < E; e++) {
        s += q_local[simd_gid][e] * k[(n_kv * S + col) * E + e];
      }
      s *= scale;
      if (!skip_mask) {
        s += mask[(n * L + row) * S + col];
      }
    }
    // If row/col fall outside the real L/S (last, partial tile), s stays
    // -INFINITY -- contributes nothing to this row's max/sum below.

    // Step C, part 1 -- online-softmax update via SIMD reduction. Every
    // thread in this row's SIMD group already holds its own score (`s`);
    // simd_max/simd_sum combine all 32 in hardware and broadcast the result
    // back to every lane, replacing the CPU version's manual loop over Bc.
    float row_max = simd_max(s);
    float new_m = max(m_i, row_max);
    float correction = exp(m_i - new_m);
    float p = exp(s - new_m);
    float row_sum = simd_sum(p);

    // Step C, part 2 -- O = P @ V. E=128 > FA_BC=32, so there aren't enough
    // threads for one-per-output-element like Step A -- instead, loop over
    // E and use simd_sum to reduce each column's 32-term dot product.
    for (int e = 0; e < E; e++) {
      float val = col_valid ? p * v[(n_kv * S + col) * E + e] : 0.0f;
      float result = simd_sum(val);
      // Every lane computes the same `correction`/`result` -- only one lane
      // should actually write, or the update would apply 32 times instead
      // of once per row, per j.
      if (simd_lid == 0) {
        o_i[simd_gid][e] = correction * o_i[simd_gid][e] + result;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    l_i = correction * l_i + row_sum;
    m_i = new_m;
  }

  // Normalize and write this tile's output -- same formula as the CPU
  // version's final step, with the 32 lanes splitting the E=128 write
  // between them instead of one thread looping over all of it.
  if (row_valid) {
    for (int e = int(simd_lid); e < E; e += FA_BC) {
      out[(n * L + row) * E + e] = o_i[simd_gid][e] / l_i;
    }
  }
}
