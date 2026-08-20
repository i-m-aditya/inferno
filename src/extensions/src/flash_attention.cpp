#include <mlx/allocator.h>
#include <mlx/ops.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/dtype.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif


namespace tiny_llm_ext {
    mx::array flash_attention(
        const mx::array &query,
        const mx::array &key,
        const mx::array &value,
        const mx::array &mask,
        const float scale,
        const bool is_causal,
        const int num_kv_heads,
        const int num_heads,
        mx::StreamOrDevice s /* = {} */
    ) {
        if (query.dtype() != mx::float32 || key.dtype() != mx::float32 || value.dtype() != mx::float32 || mask.dtype() != mx::float32) {
            throw std::runtime_error("float32 dtype expected for query");
        }

        if (num_heads % num_kv_heads != 0) {
            // throw an error
            throw std::runtime_error("hq not divisible by h");
        }

        auto out_shape  = query.shape();

        /**
         * Why does this matter here specifically?** The `mx::array` constructor's
         * third parameter is typed `std::shared_ptr<mx::Primitive>` — meaning the
         * primitive object's lifetime is *not* tied to this one function call's stack frame.
         *
         * The lazy array you're returning might get copied, stored in multiple places,
         * referenced by multiple downstream computations in the graph — and the `FlashAttention`
         * primitive backing it needs to stay alive for as long as *any* of those references still
         * exist,
         * however long that ends up being, potentially well after
         */

        return mx::array(
            out_shape,
            query.dtype(),
            std::make_shared<FlashAttention>(mlx::core::to_stream(s), scale, is_causal, num_kv_heads, num_heads),
            {query, key, value, mask}
        );


    }

    template<typename T>
    void flash_attention_impl(
        const mx::array &query, const mx::array &key, const mx::array &value, const mx::array &mask,
        mx::array &out,
        float scale_,
        bool is_causal_,
        int num_kv_heads_,
        int num_heads_,
        mx::Stream stream
    ) {

        out.set_data(mx::allocator::malloc(out.nbytes()));

        // Same encoder-registration pattern as quantized_matmul_impl: tell MLX's
        // scheduler which arrays this op reads/writes, before doing any real work.
        auto &encoder = mx::cpu::get_command_encoder(stream);
        encoder.set_input_array(query);
        encoder.set_input_array(key);
        encoder.set_input_array(value);
        encoder.set_input_array(mask);
        encoder.set_output_array(out);

        // shape of a tile
        const int Br = 32, Bc = 32;
        auto query_shape = query.shape();
        auto key_shape = key.shape();
        const int N = query_shape[0];
        const int L = query_shape[1];
        const int S = key_shape[1];
        const int E = key_shape.back();

        const int Tr = (L + Br - 1) / Br;
        const int Tc = (S + Bc - 1) / Bc;

        const int q_kv_heads_ratio = num_heads_ / num_kv_heads_; // num heads: 10, num_kv_heads: 2, ratio: 5

        // Same unsafe_weak_copy capture pattern as quantized_matmul_impl -- these
        // arrays are guaranteed to stay alive for the duration of this dispatch.
        encoder.dispatch([out_ptr = out.data<T>(), N, L, S, E, Br, Bc, Tr, Tc, is_causal_, num_kv_heads_, num_heads_,
                          q_kv_heads_ratio, scale_,
                          query = mx::array::unsafe_weak_copy(query),
                          key = mx::array::unsafe_weak_copy(key),
                          value = mx::array::unsafe_weak_copy(value),
                          mask = mx::array::unsafe_weak_copy(mask)]() {
            const T *q_ptr = query.data<T>();
            const T *k_ptr = key.data<T>();
            const T *v_ptr = value.data<T>();
            const T *mask_ptr = mask.data<T>();

            for (int n = 0; n < N; n++) {
                const int batch_idx = n / num_heads_;
                const int q_head_idx = n % num_heads_;
                const int kv_head_idx = q_head_idx / q_kv_heads_ratio;
                const int n_kv = batch_idx * num_kv_heads_ + kv_head_idx;

                for (int i = 0; i < Tr; i++) {
                    const int actual_br = std::min(Br, L - i * Br);

                    // Running state, one entry per row in this query tile -- same
                    // m/l/o you've already traced by hand, just Br of them at once.
                    std::vector<float> m_i(actual_br, -std::numeric_limits<float>::infinity());
                    std::vector<float> l_i(actual_br, 0.0f);
                    std::vector<float> o_i(actual_br * E, 0.0f);

                    for (int j = 0; j < Tc; j++) {
                        const int actual_bc = std::min(Bc, S - j * Bc);

                        // TODO (your turn): causal tile classification.
                        // Using absolute row range [i*Br, i*Br + actual_br) and column
                        // range [j*Bc, j*Bc + actual_bc), and the same offset rule as
                        // causal_mask (valid iff col <= row + (S - L)):
                        //   - if is_causal_ and every (row, col) pair in this tile is
                        //     invalid -> `continue;` (skip straight to the next j)
                        //   - if is_causal_ and every pair is valid -> remember this
                        //     (e.g. a bool), so Step B below skips reading mask_ptr
                        //   - otherwise -> this tile straddles the diagonal, needs the
                        //     full masked path

                        auto row = std::pair(i*Br, i*Br + actual_br -1);
                        auto col = std::pair(j*Bc, j*Bc + actual_bc -1);
                        int offset = S - L;
                        bool skip_mask = false;
                        if (is_causal_) {
                            if (col.first > (row.second  + offset)) {
                                continue;
                            } else if (col.second <= row.first + offset) {
                                skip_mask = true;
                            }
                        }

                        std::vector<float> scores(actual_br * actual_bc);

                        // for one tile, doing q*K^T and then scaling
                        for (int r=0; r < actual_br; r++) {
                            for (int c=0; c < actual_bc; c++) {
                                auto q_row = q_ptr + (n*L + i*Br + r) * E;
                                auto k_row = k_ptr + (n_kv*S + j*Bc + c) * E;
                                float sum = 0;
                                for (int e=0; e<E; e++) {
                                    sum += q_row[e] * k_row[e];
                                }
                                scores[r*actual_bc + c] = sum*scale_;

                                if (!skip_mask) {
                                    scores[r*actual_bc + c] += mask_ptr[(n*L + i*Br + r)*S + (j*Bc+c)];
                                }

                            }
                        }

                        for (int r=0; r<actual_br; r++) {
                            float local_max = -std::numeric_limits<float>::infinity();
                            for (int c=0; c<actual_bc; c++) {
                                local_max = fmax(local_max, scores[r*actual_bc + c]);
                            }
                            float new_m = fmax(m_i[r], local_max);
                            float correction = exp(m_i[r] - new_m);

                            // rescale the existing accumulated output by the correction
                            // factor before adding this tile's contribution
                            for (int e=0; e<E; e++) {
                                o_i[r*E + e] *= correction;
                            }

                            float new_l = correction * l_i[r];
                            for (int c=0; c<actual_bc; c++) {
                                // computed once per (r,c), reused for both l and o below
                                float p = exp(scores[r*actual_bc + c] - new_m);
                                new_l += p;
                                for (int e=0; e<E; e++) {
                                    o_i[r*E + e] += p * v_ptr[(n_kv*S + j*Bc + c)*E + e];
                                }
                            }

                            m_i[r] = new_m;
                            l_i[r] = new_l;
                        }
                    }


                    for (int r=0; r<actual_br; r++) {
                        for (int e=0; e<E; e++) {
                            out_ptr[(n*L + i*Br + r) * E + e] = static_cast<float>(o_i[r*E+e] / l_i[r]);
                        }
                    }
                }
            }
        });
    }

    void FlashAttention::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
        auto &query = inputs[0];
        auto &key = inputs[1];
        auto &value = inputs[2];
        auto &mask = inputs[3];

        auto &out = outputs[0];


        if (out.dtype() == mlx::core::float32) {
            return flash_attention_impl<float>(
                query,
                key,
                value,
                mask,
                out,
                scale_,
                is_causal_,
                num_kv_heads_,
                num_heads_,
                stream()
            );
        } else {
            throw std::runtime_error("Only float32 is supported");
        }

    }

    void FlashAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
        auto &query = inputs[0];
        auto &key = inputs[1];
        auto &value = inputs[2];
        auto &mask = inputs[3];
        auto &out = outputs[0];

        // Same contiguity requirement as the CPU path -- the kernel indexes
        // these with simple raw offsets, no stride handling.
        if (!query.flags().row_contiguous || !key.flags().row_contiguous ||
            !value.flags().row_contiguous || !mask.flags().row_contiguous) {
            throw std::runtime_error("flash_attention: all inputs must be contiguous");
        }

        auto &s = stream();
        auto &d = mx::metal::device(s.device);

        out.set_data(mx::allocator::malloc(out.nbytes()));

        auto query_shape = query.shape();
        auto key_shape = key.shape();
        const int N = query_shape[0];
        const int L = query_shape[1];
        const int S = key_shape[1];
        const int E = key_shape.back();
        if (E > 128) {
            // The kernel's threadgroup memory is sized for a max of E=128
            // (compile-time size) -- see flash_attention.metal. E itself is
            // a runtime value; anything up to 128 works.
            throw std::runtime_error("flash_attention GPU kernel only supports E <= 128");
        }

        const int Br = 32;
        const int Tr = (L + Br - 1) / Br;

        auto library = d.get_library("tiny_llm_ext");
        auto kernel = d.get_kernel("flash_attention_f32_e128", library);

        auto &compute_encoder = d.get_command_encoder(s.index);
        compute_encoder.set_compute_pipeline_state(kernel);

        // Buffer indices here must match [[buffer(N)]] in flash_attention.metal.
        compute_encoder.set_input_array(query, 0);
        compute_encoder.set_input_array(key, 1);
        compute_encoder.set_input_array(value, 2);
        compute_encoder.set_input_array(mask, 3);
        compute_encoder.set_output_array(out, 4);
        compute_encoder.set_bytes(N, 5);
        compute_encoder.set_bytes(L, 6);
        compute_encoder.set_bytes(S, 7);
        compute_encoder.set_bytes(E, 8);
        compute_encoder.set_bytes(num_kv_heads_, 9);
        compute_encoder.set_bytes(num_heads_, 10);
        compute_encoder.set_bytes(scale_, 11);
        compute_encoder.set_bytes(is_causal_, 12);

        // One threadgroup per (n, i); dispatch_threadgroups takes the
        // threadgroup COUNT directly (unlike dispatch_threads, which takes a
        // total thread count and computes the threadgroup count itself).
        MTL::Size grid_dims = MTL::Size(N, Tr, 1);
        MTL::Size group_dims = MTL::Size(32, 32, 1);

        compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    }
}
