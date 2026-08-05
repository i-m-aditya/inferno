#pragma once

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
}  // namespace tiny_llm_ext
