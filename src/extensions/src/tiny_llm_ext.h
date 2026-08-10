#pragma once

#include <mlx/array.h>
#include <mlx/utils.h>
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace tiny_llm_ext {

    void load_library(mx::Device d, const char *path);

    mx::array quantized_matmul(const mx::array &scales,   // Input array scales
                            const mx::array &biases,   // Input array biases
                            const int group_size,      // Group size
                            const int bits,            // Number of bits
                            const mx::array &a,        // Input array a (not quantized)
                            const mx::array &b,        // Input array b (quantized)
                            const bool transpose_b,    // Whether to transpose b
                            mx::StreamOrDevice s = {}  // Stream on which to schedule the operation
    );

    class QuantizedMatmul : public mx::Primitive {
    public:
        explicit QuantizedMatmul(mx::Stream stream) : mx::Primitive(stream) {};

        void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
        void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;

        std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &inputs,
                                                                const std::vector<int> &axes) override {
            throw std::runtime_error("QuantizedMatmul has no vmap implementation.");
        }

        const char *name() const override { return "QuantizedMatmul"; }
    };

    mx::array flash_attention(
    const mx::array &query,
    const mx::array &key,
    const mx::array &value,
    const mx::array &mask,
    const float scale,
    const bool is_causal,
    const int num_kv_heads,
    const int num_heads,
    mx::StreamOrDevice s = {}
    );


    class FlashAttention : public mx::Primitive {
        public:
            explicit FlashAttention(mx::Stream stream, float scale, bool is_causal, int num_kv_heads, int num_heads): mx::Primitive(stream), scale_(scale), is_causal_(is_causal), num_kv_heads_(num_kv_heads), num_heads_(num_heads){};


            void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
            void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;


            std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &inputs,
                                                                    const std::vector<int> &axes) override {
                throw std::runtime_error("FlashAttention has no vmap implementation.");
            }

            const char *name() const override { return "FlashAttention"; }

        private:
            float scale_;
            bool is_causal_;
            int num_kv_heads_;
            int num_heads_;


    };

}  // namespace tiny_llm_ext
