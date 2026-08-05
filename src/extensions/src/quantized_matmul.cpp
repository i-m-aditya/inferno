#include <cstdint>
#include <stdexcept>

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
    mx::array quantized_matmul(const mx::array &scales,         // Input array scales
                               const mx::array &biases,         // Input array biases
                               const int group_size,            // Group size
                               const int bits,                  // Number of bits
                               const mx::array &a,              // Input array a (not quantized)
                               const mx::array &b,              // Input array b (quantized)
                               const bool transpose_b,          // Whether to transpose b
                               mx::StreamOrDevice s /* = {} */  // Stream on which to schedule the operation
    ) {
        if (scales.dtype() != mx::float16 && scales.dtype() != mx::bfloat16) {
            throw std::runtime_error("quantized_matmul: scales must be float16 or bfloat16");
        }
        if (scales.dtype() != biases.dtype()) {
            throw std::runtime_error("quantized_matmul: scales and biases must be the same dtype");
        }
        if (b.dtype() != mx::uint32) {
            throw std::runtime_error("quantized_matmul: b must be uint32");
        }
        if (a.dtype() != scales.dtype()) {
            throw std::runtime_error("quantized_matmul: a must be the same dtype as scales");
        }
        if (a.shape().size() != 2) {
            throw std::runtime_error("quantized_matmul: a must be a 2D array");
        }
        if (b.shape().size() != 2) {
            throw std::runtime_error("quantized_matmul: b must be a 2D array");
        }
        if (bits != 4) {
            throw std::runtime_error("quantized_matmul: bits must be 4");
        }
        const int packs_per_item = 32 / bits;
        if (group_size != 128) {
            throw std::runtime_error("quantized_matmul: group_size must be 128");
        }
        auto out_shape = a.shape();
        if (out_shape.size() != 2) {
            throw std::runtime_error("quantized_matmul: a must be a 2D array");
        }
        out_shape[1] = b.shape()[0];
        if (!transpose_b) {
            throw std::runtime_error("quantized_matmul: b must be transposed");
        }

        if (scales.shape() != biases.shape()) {
            throw std::runtime_error("quantized_matmul: scales and biases must have the same shape");
        }
        if (b.shape()[0] != scales.shape()[0]) {
            throw std::runtime_error("quantized_matmul: b must have the same number of rows as scales");
        }
        if (a.shape()[1] % group_size != 0) {
            throw std::runtime_error("quantized_matmul: a columns must be divisible by group_size");
        }
        if (scales.shape()[1] != a.shape()[1] / group_size) {
            throw std::runtime_error("quantized_matmul: scales must have one column per input group");
        }
        if (b.shape()[1] != a.shape()[1] / packs_per_item) {
            throw std::runtime_error("quantized_matmul: a must have the same number of columns as b");
        }

        return mx::array(
            /* const mx::Shape& shape = */ out_shape,
            /* mx::Dtype dtype = */ a.dtype(),
            /* std::shared_ptr<mx::Primitive> primitive = */
            std::make_shared<QuantizedMatmul>(to_stream(s)),
            /* const std::vector<mx::array>& inputs = */ {scales, biases, a, b});
    }

    template <typename T>
    void quantized_matmul_impl(const mx::array &scales, const mx::array &biases, const mx::array &a,
                                const mx::array &b, mx::array &out, mx::Stream stream) {
        out.set_data(mx::allocator::malloc(out.nbytes()));

        auto &encoder = mx::cpu::get_command_encoder(stream);
        encoder.set_input_array(scales);
        encoder.set_input_array(biases);
        encoder.set_input_array(a);
        encoder.set_input_array(b);
        encoder.set_output_array(out);

        if (!a.flags().row_contiguous) {
            throw std::runtime_error("quantized_matmul: a must be contiguous");
        }
        if (!b.flags().row_contiguous) {
            throw std::runtime_error("quantized_matmul: b must be contiguous");
        }

        // unsafe_weak_copy: lets the lambda below capture these arrays without
        // bumping their refcount (they're guaranteed to stay alive for the
        // duration of this dispatch) -- same pattern used elsewhere for
        // encoder.dispatch lambdas.
        encoder.dispatch([out_ptr = out.data<T>(), out_shape = out.shape(), out_strides = out.strides(),
                          a = mx::array::unsafe_weak_copy(a), b = mx::array::unsafe_weak_copy(b),
                          scales = mx::array::unsafe_weak_copy(scales),
                          biases = mx::array::unsafe_weak_copy(biases)]() {
            int M = a.shape()[0];
            int N = a.shape()[1];
            int K = b.shape()[0];
            const int group_size = 128;
            const int bits = 4;
            const int group_per_row = N / group_size;
            const T *a_ptr = a.data<T>();
            const uint32_t *b_ptr = b.data<uint32_t>();
            const T *scales_ptr = scales.data<T>();
            const T *biases_ptr = biases.data<T>();
            uint32_t item_mask = (1 << bits) - 1;

            for (int i = 0; i < M; i++) {
                for (int k = 0; k < K; k++) {
                    float sum = 0;
                    for (int group_idx = 0; group_idx < group_per_row; group_idx++) {
                        int64_t scales_loc =
                            mx::elem_to_loc(k * group_per_row + group_idx, scales.shape(), scales.strides());
                        int64_t biases_loc =
                            mx::elem_to_loc(k * group_per_row + group_idx, biases.shape(), biases.strides());
                        T scale = scales_ptr[scales_loc];
                        T bias = biases_ptr[biases_loc];

                        // TODO (your turn): unpack this one group's `group_size` (128) quantized
                        // values from `b_ptr`, dequantize each, multiply by the matching `a_ptr`
                        // value, and add the result into `sum`.
                        //
                        // Facts to use:
                        //   - `b` is stored row-major as (K, N/8) uint32s. Row k's packed values
                        //     for this group start at flat index: k * (N/8) + group_idx * (group_size/8)
                        //   - Each uint32 packs 8 consecutive 4-bit values (group_size/8 = 16 uint32s
                        //     per group). To read the j-th value (j in 0..7) out of one packed uint32:
                        //         quantized = (packed >> (j * bits)) & item_mask
                        //   - The matching activation value for the p-th value in this group (p in
                        //     0..group_size-1, counting across all 16 packs) is:
                        //         a_ptr[i * N + group_idx * group_size + p]
                        //   - Dequantize: dequantized = (float)quantized * (float)scale + (float)bias
                        //   - Accumulate: sum += (float)a_value * dequantized
                        //

                        int64_t b_loc = mx::elem_to_loc((k * N + group_idx * group_size)/8, b.shape(), b.strides());
                        int64_t a_loc = mx::elem_to_loc((i*N) + group_idx * group_size, a.shape(), a.strides());

                        const int packs_per_item = 32 / bits;

                        for (int item_idx = 0; item_idx < group_size; item_idx += packs_per_item) {
                            uint32_t b_val = b_ptr[b_loc];
                            uint8_t *b_bytes = reinterpret_cast<uint8_t *>(&b_val);

                            for (int pack_idx = 0; pack_idx < packs_per_item; pack_idx ++) {
                                // this line is very smart, `pack_idx%2` woudl be 0,1, and so ((pack_idx%2) * bits)) would be 0 or 4
                                // and hence, we can last and first 4 bits repectively, even though we dont have primitive type uint4
                                uint8_t item_value = (b_bytes[pack_idx / 2] >> ((pack_idx%2) * bits)) & item_mask;
                                float b = static_cast<float>(item_value) * static_cast<float>(scale) + static_cast<float>(bias);
                                float a = static_cast<float>(a_ptr[a_loc]);
                                sum += a*b;
                                a_loc += 1;
                            }

                            b_loc += 1;

                        }
                    }
                    int64_t out_loc = mx::elem_to_loc(i * K + k, out_shape, out_strides);
                    out_ptr[out_loc] = static_cast<T>(sum);
                }
            }
        });
    }

    void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
        auto &scales = inputs[0];
        auto &biases = inputs[1];
        auto &a = inputs[2];
        auto &b = inputs[3];
        auto &out = outputs[0];

        if (out.dtype() == mx::float16) {
            return quantized_matmul_impl<mx::float16_t>(scales, biases, a, b, out, stream());
        } else if (out.dtype() == mx::bfloat16) {
            return quantized_matmul_impl<mx::bfloat16_t>(scales, biases, a, b, out, stream());
        } else {
            throw std::runtime_error("quantized_matmul: unsupported dtype for eval_cpu");
        }
    }

    void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
        auto &scales = inputs[0];
        auto &biases = inputs[1];
        auto &a = inputs[2];
        auto &b = inputs[3];
        auto &out = outputs[0];

        // Same restriction as the CPU path -- simple raw pointer-style indexing
        // in the kernel only works if these are laid out row-major with no gaps.
        if (!a.flags().row_contiguous) {
            throw std::runtime_error("quantized_matmul: a must be contiguous");
        }
        if (!b.flags().row_contiguous) {
            throw std::runtime_error("quantized_matmul: b must be contiguous");
        }

        if (!scales.flags().row_contiguous) {
            throw std::runtime_error("quantized_matmul: scales must be contiguous");
        }


        if (!biases.flags().row_contiguous) {
            throw std::runtime_error("quantized_matmul: scales must be contiguous");
        }


        auto &s = stream();
        auto &d = mx::metal::device(s.device);

        out.set_data(mx::allocator::malloc(out.nbytes()));

        int M = a.shape()[0];
        int N = a.shape()[1];
        int K = b.shape()[0];

        // Resolve which compiled kernel variant to use (matches the names produced
        // by `instantiate_quantized_matmul` in quantized_matmul.metal).
        std::ostringstream kname;
        kname << "quantized_matmul_" << type_to_name(out);

        auto library = d.get_library("tiny_llm_ext");
        auto kernel = d.get_kernel(kname.str(), library);

        auto &compute_encoder = d.get_command_encoder(s.index);
        compute_encoder.set_compute_pipeline_state(kernel);

        // Buffer indices here must match [[buffer(N)]] in quantized_matmul.metal exactly.
        compute_encoder.set_input_array(a, 0);
        compute_encoder.set_input_array(b, 1);
        compute_encoder.set_input_array(scales, 2);
        compute_encoder.set_input_array(biases, 3);
        compute_encoder.set_output_array(out, 4);
        compute_encoder.set_bytes(M, 5);
        compute_encoder.set_bytes(N, 6);
        compute_encoder.set_bytes(K, 7);

        // 2D grid: one thread per output element, x-axis over K, y-axis over M
        // (has to match how the kernel reads index.x / index.y).
        size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
        size_t group_m = std::min<size_t>(M, 32);
        size_t group_k = std::min<size_t>(K, std::max<size_t>(1, max_threads / group_m));

        MTL::Size group_dims = MTL::Size(group_k, group_m, 1);
        MTL::Size grid_dims = MTL::Size(K, M, 1);

        compute_encoder.dispatch_threads(grid_dims, group_dims);
    }
}
