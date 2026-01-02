/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Adam Gibson
//

#ifndef DEV_TESTS_LLAMACPPUTILS_H
#define DEV_TESTS_LLAMACPPUTILS_H

#include <array/NDArray.h>
#include <graph/Context.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#if HAVE_LLAMACPP
#include <llama.h>
#include <ggml.h>
#endif

using namespace samediff;

namespace sd {
namespace ops {
namespace platforms {

/**
 * Platform helper declarations for llama.cpp accelerated operations
 */

//////////////////////////////////////////////////////////////////////////
// Core LLM Operations - CPU
//////////////////////////////////////////////////////////////////////////

// Matrix multiplication optimized with GGML kernels
DECLARE_PLATFORM(matmul, ENGINE_CPU);

// Quantized matrix multiplication (Q4_0, Q4_1, Q8_0, etc.)
DECLARE_PLATFORM(quantized_matmul, ENGINE_CPU);

// RMS normalization (used in LLaMA models)
DECLARE_PLATFORM(rms_norm, ENGINE_CPU);

// RoPE (Rotary Position Embedding)
DECLARE_PLATFORM(rope, ENGINE_CPU);

// SiLU activation (Swish variant used in LLaMA)
DECLARE_PLATFORM(silu, ENGINE_CPU);

// Grouped Query Attention
DECLARE_PLATFORM(grouped_query_attention, ENGINE_CPU);

// Flash Attention (if supported by GGML backend)
DECLARE_PLATFORM(flash_attention, ENGINE_CPU);

// Softmax optimized with GGML
DECLARE_PLATFORM(softmax, ENGINE_CPU);

// GELU activation
DECLARE_PLATFORM(gelu, ENGINE_CPU);

// Layer normalization
DECLARE_PLATFORM(layer_norm, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// Core LLM Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(matmul, ENGINE_CUDA);
DECLARE_PLATFORM(quantized_matmul, ENGINE_CUDA);
DECLARE_PLATFORM(rms_norm, ENGINE_CUDA);
DECLARE_PLATFORM(rope, ENGINE_CUDA);
DECLARE_PLATFORM(silu, ENGINE_CUDA);
DECLARE_PLATFORM(grouped_query_attention, ENGINE_CUDA);
DECLARE_PLATFORM(flash_attention, ENGINE_CUDA);
DECLARE_PLATFORM(softmax, ENGINE_CUDA);
DECLARE_PLATFORM(gelu, ENGINE_CUDA);
DECLARE_PLATFORM(layer_norm, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// Elementwise Math Operations - CPU
//////////////////////////////////////////////////////////////////////////

// Absolute value
DECLARE_PLATFORM(abs, ENGINE_CPU);

// Negation
DECLARE_PLATFORM(neg, ENGINE_CPU);

// Square
DECLARE_PLATFORM(square, ENGINE_CPU);

// Square root
DECLARE_PLATFORM(sqrt, ENGINE_CPU);

// Exponential
DECLARE_PLATFORM(exp, ENGINE_CPU);

// Natural logarithm
DECLARE_PLATFORM(log, ENGINE_CPU);

// Sine
DECLARE_PLATFORM(sin, ENGINE_CPU);

// Cosine
DECLARE_PLATFORM(cos, ENGINE_CPU);

// Hyperbolic tangent
DECLARE_PLATFORM(tanh, ENGINE_CPU);

// Ceiling
DECLARE_PLATFORM(ceil, ENGINE_CPU);

// Floor
DECLARE_PLATFORM(floor, ENGINE_CPU);

// Round to nearest
DECLARE_PLATFORM(round, ENGINE_CPU);

// Sign function
DECLARE_PLATFORM(sign, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// Elementwise Math Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(abs, ENGINE_CUDA);
DECLARE_PLATFORM(neg, ENGINE_CUDA);
DECLARE_PLATFORM(square, ENGINE_CUDA);
DECLARE_PLATFORM(sqrt, ENGINE_CUDA);
DECLARE_PLATFORM(exp, ENGINE_CUDA);
DECLARE_PLATFORM(log, ENGINE_CUDA);
DECLARE_PLATFORM(sin, ENGINE_CUDA);
DECLARE_PLATFORM(cos, ENGINE_CUDA);
DECLARE_PLATFORM(tanh, ENGINE_CUDA);
DECLARE_PLATFORM(ceil, ENGINE_CUDA);
DECLARE_PLATFORM(floor, ENGINE_CUDA);
DECLARE_PLATFORM(round, ENGINE_CUDA);
DECLARE_PLATFORM(sign, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// Activation Functions - CPU
//////////////////////////////////////////////////////////////////////////

// Rectified Linear Unit
DECLARE_PLATFORM(relu, ENGINE_CPU);

// Exponential Linear Unit
DECLARE_PLATFORM(elu, ENGINE_CPU);

// Sigmoid
DECLARE_PLATFORM(sigmoid, ENGINE_CPU);

// Hard Swish
DECLARE_PLATFORM(hardswish, ENGINE_CPU);

// Hard Sigmoid
DECLARE_PLATFORM(hardsigmoid, ENGINE_CPU);

// Leaky ReLU
DECLARE_PLATFORM(leakyrelu, ENGINE_CPU);

// Softplus
DECLARE_PLATFORM(softplus, ENGINE_CPU);

// Step function
DECLARE_PLATFORM(step, ENGINE_CPU);

// GELU (using error function approximation)
DECLARE_PLATFORM(gelu_erf, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// Activation Functions - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(relu, ENGINE_CUDA);
DECLARE_PLATFORM(elu, ENGINE_CUDA);
DECLARE_PLATFORM(sigmoid, ENGINE_CUDA);
DECLARE_PLATFORM(hardswish, ENGINE_CUDA);
DECLARE_PLATFORM(hardsigmoid, ENGINE_CUDA);
DECLARE_PLATFORM(leakyrelu, ENGINE_CUDA);
DECLARE_PLATFORM(softplus, ENGINE_CUDA);
DECLARE_PLATFORM(step, ENGINE_CUDA);
DECLARE_PLATFORM(gelu_erf, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// Tensor Manipulation Operations - CPU
//////////////////////////////////////////////////////////////////////////

// Concatenate tensors
DECLARE_PLATFORM(concat, ENGINE_CPU);

// Pad tensor
DECLARE_PLATFORM(pad, ENGINE_CPU);

// Repeat tensor
DECLARE_PLATFORM(repeat, ENGINE_CPU);

// Upsampling 2D
DECLARE_PLATFORM(upsampling2d, ENGINE_CPU);

// Gather elements
DECLARE_PLATFORM(gather, ENGINE_CPU);

// Diagonal matrix
DECLARE_PLATFORM(diag, ENGINE_CPU);

// Transpose tensor
DECLARE_PLATFORM(transpose, ENGINE_CPU);

// Permute dimensions
DECLARE_PLATFORM(permute, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// Tensor Manipulation Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(concat, ENGINE_CUDA);
DECLARE_PLATFORM(pad, ENGINE_CUDA);
DECLARE_PLATFORM(repeat, ENGINE_CUDA);
DECLARE_PLATFORM(upsampling2d, ENGINE_CUDA);
DECLARE_PLATFORM(gather, ENGINE_CUDA);
DECLARE_PLATFORM(diag, ENGINE_CUDA);
DECLARE_PLATFORM(transpose, ENGINE_CUDA);
DECLARE_PLATFORM(permute, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// Convolution and Pooling Operations - CPU
//////////////////////////////////////////////////////////////////////////

// 2D Convolution
DECLARE_PLATFORM(conv2d, ENGINE_CPU);

// Depthwise 2D Convolution
DECLARE_PLATFORM(depthwise_conv2d, ENGINE_CPU);

// Max Pooling 2D
DECLARE_PLATFORM(maxpool2d, ENGINE_CPU);

// Average Pooling 2D
DECLARE_PLATFORM(avgpool2d, ENGINE_CPU);

// Image to Column transformation
DECLARE_PLATFORM(im2col, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// Convolution and Pooling Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(conv2d, ENGINE_CUDA);
DECLARE_PLATFORM(depthwise_conv2d, ENGINE_CUDA);
DECLARE_PLATFORM(maxpool2d, ENGINE_CUDA);
DECLARE_PLATFORM(avgpool2d, ENGINE_CUDA);
DECLARE_PLATFORM(im2col, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// Advanced Operations - CPU
//////////////////////////////////////////////////////////////////////////

// Argsort
DECLARE_PLATFORM(argsort, ENGINE_CPU);

// Argmax
DECLARE_PLATFORM(argmax, ENGINE_CPU);

// Top-K elements
DECLARE_PLATFORM(top_k, ENGINE_CPU);

// Cumulative sum
DECLARE_PLATFORM(cumsum, ENGINE_CPU);

// Clip by value (clamp)
DECLARE_PLATFORM(clip_by_value, ENGINE_CPU);

// Range (arange)
DECLARE_PLATFORM(range, ENGINE_CPU);

// Scalar multiplication
DECLARE_PLATFORM(mul_scalar, ENGINE_CPU);

// Reduce mean
DECLARE_PLATFORM(reduce_mean, ENGINE_CPU);

// Count equal elements
DECLARE_PLATFORM(count_equal, ENGINE_CPU);

// Diagonal mask infinity
DECLARE_PLATFORM(diag_mask_inf, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// Advanced Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(argsort, ENGINE_CUDA);
DECLARE_PLATFORM(argmax, ENGINE_CUDA);
DECLARE_PLATFORM(top_k, ENGINE_CUDA);
DECLARE_PLATFORM(cumsum, ENGINE_CUDA);
DECLARE_PLATFORM(clip_by_value, ENGINE_CUDA);
DECLARE_PLATFORM(range, ENGINE_CUDA);
DECLARE_PLATFORM(mul_scalar, ENGINE_CUDA);
DECLARE_PLATFORM(reduce_mean, ENGINE_CUDA);
DECLARE_PLATFORM(count_equal, ENGINE_CUDA);
DECLARE_PLATFORM(diag_mask_inf, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// Normalization Operations - CPU
//////////////////////////////////////////////////////////////////////////

// Group normalization
DECLARE_PLATFORM(group_norm, ENGINE_CPU);

// L2 normalization
DECLARE_PLATFORM(l2_normalize, ENGINE_CPU);

// Standardize (mean/variance normalization)
DECLARE_PLATFORM(standardize, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// Normalization Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(group_norm, ENGINE_CUDA);
DECLARE_PLATFORM(l2_normalize, ENGINE_CUDA);
DECLARE_PLATFORM(standardize, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// Binary Operations - CPU
//////////////////////////////////////////////////////////////////////////

// Element-wise addition
DECLARE_PLATFORM(add, ENGINE_CPU);

// Element-wise subtraction
DECLARE_PLATFORM(subtract, ENGINE_CPU);

// Element-wise multiplication
DECLARE_PLATFORM(multiply, ENGINE_CPU);

// Element-wise division
DECLARE_PLATFORM(divide, ENGINE_CPU);

// Tensor multiplication (outer product)
DECLARE_PLATFORM(tensormmul, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// Binary Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(add, ENGINE_CUDA);
DECLARE_PLATFORM(subtract, ENGINE_CUDA);
DECLARE_PLATFORM(multiply, ENGINE_CUDA);
DECLARE_PLATFORM(divide, ENGINE_CUDA);
DECLARE_PLATFORM(tensormmul, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// LLM-Specific Operations - CPU
//////////////////////////////////////////////////////////////////////////

// Timestep embedding for diffusion models
DECLARE_PLATFORM(timestep_embedding, ENGINE_CPU);

// ReGLU (ReLU Gated Linear Unit)
DECLARE_PLATFORM(reglu, ENGINE_CPU);

// SwiGLU (Swish Gated Linear Unit)
DECLARE_PLATFORM(swiglu, ENGINE_CPU);

// Gated Linear Attention
DECLARE_PLATFORM(gated_linear_attn, ENGINE_CPU);

// Fill tensor with constant value
DECLARE_PLATFORM(fill, ENGINE_CPU);

// exp(x) - 1 (accurate for small x)
DECLARE_PLATFORM(expm1, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// LLM-Specific Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(timestep_embedding, ENGINE_CUDA);
DECLARE_PLATFORM(reglu, ENGINE_CUDA);
DECLARE_PLATFORM(swiglu, ENGINE_CUDA);
DECLARE_PLATFORM(gated_linear_attn, ENGINE_CUDA);
DECLARE_PLATFORM(fill, ENGINE_CUDA);
DECLARE_PLATFORM(expm1, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// Backward/Gradient Operations - CPU
//////////////////////////////////////////////////////////////////////////

// SiLU backward pass
DECLARE_PLATFORM(silu_bp, ENGINE_CPU);

// RMS normalization backward pass
DECLARE_PLATFORM(rms_norm_bp, ENGINE_CPU);

// RoPE backward pass
DECLARE_PLATFORM(rope_bp, ENGINE_CPU);

// Softmax backward pass
DECLARE_PLATFORM(softmax_bp, ENGINE_CPU);

// Max pooling 2D backward pass
DECLARE_PLATFORM(maxpool2d_bp, ENGINE_CPU);

// Average pooling 2D backward pass
DECLARE_PLATFORM(avgpool2d_bp, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// Backward/Gradient Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(silu_bp, ENGINE_CUDA);
DECLARE_PLATFORM(rms_norm_bp, ENGINE_CUDA);
DECLARE_PLATFORM(rope_bp, ENGINE_CUDA);
DECLARE_PLATFORM(softmax_bp, ENGINE_CUDA);
DECLARE_PLATFORM(maxpool2d_bp, ENGINE_CUDA);
DECLARE_PLATFORM(avgpool2d_bp, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// Tensor View/Copy Operations - CPU
//////////////////////////////////////////////////////////////////////////

// Duplicate tensor
DECLARE_PLATFORM(dup, ENGINE_CPU);

// Make tensor contiguous
DECLARE_PLATFORM(contiguous, ENGINE_CPU);

// Reshape tensor
DECLARE_PLATFORM(reshape, ENGINE_CPU);

// Copy tensor
DECLARE_PLATFORM(copy, ENGINE_CPU);

// Assign tensor
DECLARE_PLATFORM(assign, ENGINE_CPU);

// Flatten tensor
DECLARE_PLATFORM(flatten, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// Tensor View/Copy Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(dup, ENGINE_CUDA);
DECLARE_PLATFORM(contiguous, ENGINE_CUDA);
DECLARE_PLATFORM(reshape, ENGINE_CUDA);
DECLARE_PLATFORM(copy, ENGINE_CUDA);
DECLARE_PLATFORM(assign, ENGINE_CUDA);
DECLARE_PLATFORM(flatten, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// Reduction Operations - CPU
//////////////////////////////////////////////////////////////////////////

// Sum all elements
DECLARE_PLATFORM(reduce_sum, ENGINE_CPU);

// Sum along rows
DECLARE_PLATFORM(reduce_sum_rows, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// Reduction Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(reduce_sum, ENGINE_CUDA);
DECLARE_PLATFORM(reduce_sum_rows, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// 1D Convolution and Pooling Operations - CPU
//////////////////////////////////////////////////////////////////////////

// 1D Convolution
DECLARE_PLATFORM(conv1d, ENGINE_CPU);

// 1D Max Pooling
DECLARE_PLATFORM(maxpool1d, ENGINE_CPU);

// 1D Average Pooling
DECLARE_PLATFORM(avgpool1d, ENGINE_CPU);

// 1D Transposed Convolution (Deconvolution)
DECLARE_PLATFORM(deconv1d, ENGINE_CPU);

// 1D Max Pooling backward pass
DECLARE_PLATFORM(maxpool1d_bp, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// 1D Convolution and Pooling Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(conv1d, ENGINE_CUDA);
DECLARE_PLATFORM(maxpool1d, ENGINE_CUDA);
DECLARE_PLATFORM(avgpool1d, ENGINE_CUDA);
DECLARE_PLATFORM(deconv1d, ENGINE_CUDA);
DECLARE_PLATFORM(maxpool1d_bp, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// State Space Model Operations (Mamba) - CPU
//////////////////////////////////////////////////////////////////////////

// SSM Convolution
DECLARE_PLATFORM(ssm_conv, ENGINE_CPU);

// SSM Selective Scan
DECLARE_PLATFORM(ssm_scan, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// State Space Model Operations (Mamba) - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(ssm_conv, ENGINE_CUDA);
DECLARE_PLATFORM(ssm_scan, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// Loss Function Operations - CPU
//////////////////////////////////////////////////////////////////////////

// Softmax cross entropy loss
DECLARE_PLATFORM(softmax_cross_entropy_loss, ENGINE_CPU);

// Softmax cross entropy loss backward pass
DECLARE_PLATFORM(softmax_cross_entropy_loss_bp, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// Loss Function Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(softmax_cross_entropy_loss, ENGINE_CUDA);
DECLARE_PLATFORM(softmax_cross_entropy_loss_bp, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// RWKV Operations - CPU
//////////////////////////////////////////////////////////////////////////

// RWKV v6 Weighted Key-Value
DECLARE_PLATFORM(rwkv_wkv6, ENGINE_CPU);

// RWKV v7 Weighted Key-Value
DECLARE_PLATFORM(rwkv_wkv7, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// RWKV Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(rwkv_wkv6, ENGINE_CUDA);
DECLARE_PLATFORM(rwkv_wkv7, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// Window Operations (Vision Transformers) - CPU
//////////////////////////////////////////////////////////////////////////

// Window partition (Swin Transformer)
DECLARE_PLATFORM(win_part, ENGINE_CPU);

// Window unpartition (Swin Transformer)
DECLARE_PLATFORM(win_unpart, ENGINE_CPU);

// GeGLU (Gated GELU) activation
DECLARE_PLATFORM(geglu, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// Window Operations (Vision Transformers) - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(win_part, ENGINE_CUDA);
DECLARE_PLATFORM(win_unpart, ENGINE_CUDA);
DECLARE_PLATFORM(geglu, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// Embedding Operations - CPU
//////////////////////////////////////////////////////////////////////////

// Embedding lookup from weight matrix
DECLARE_PLATFORM(embedding_lookup, ENGINE_CPU);

// Embedding lookup backward
DECLARE_PLATFORM(embedding_lookup_bp, ENGINE_CPU);

// Sinusoidal position encoding
DECLARE_PLATFORM(sinusoidal_position_encoding, ENGINE_CPU);

// ALiBi position bias
DECLARE_PLATFORM(alibi_position_bias, ENGINE_CPU);

// Get rows from matrix
DECLARE_PLATFORM(get_rows, ENGINE_CPU);

// Get rows backward
DECLARE_PLATFORM(get_rows_bp, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// Embedding Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(embedding_lookup, ENGINE_CUDA);
DECLARE_PLATFORM(embedding_lookup_bp, ENGINE_CUDA);
DECLARE_PLATFORM(sinusoidal_position_encoding, ENGINE_CUDA);
DECLARE_PLATFORM(alibi_position_bias, ENGINE_CUDA);
DECLARE_PLATFORM(get_rows, ENGINE_CUDA);
DECLARE_PLATFORM(get_rows_bp, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// Quantization Operations - CPU
//////////////////////////////////////////////////////////////////////////

// Quantize to Q4_0 format
DECLARE_PLATFORM(quantize_q4_0, ENGINE_CPU);

// Quantize to Q8_0 format
DECLARE_PLATFORM(quantize_q8_0, ENGINE_CPU);

// Dequantize to float
DECLARE_PLATFORM(dequantize, ENGINE_CPU);

// Quantized matrix multiplication
DECLARE_PLATFORM(quantized_mul_mat, ENGINE_CPU);

// Scale tensor
DECLARE_PLATFORM(scale, ENGINE_CPU);

// In-place addition
DECLARE_PLATFORM(add_inplace, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// Quantization Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(quantize_q4_0, ENGINE_CUDA);
DECLARE_PLATFORM(quantize_q8_0, ENGINE_CUDA);
DECLARE_PLATFORM(dequantize, ENGINE_CUDA);
DECLARE_PLATFORM(quantized_mul_mat, ENGINE_CUDA);
DECLARE_PLATFORM(scale, ENGINE_CUDA);
DECLARE_PLATFORM(add_inplace, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// Mixture of Experts Operations - CPU
//////////////////////////////////////////////////////////////////////////

// MoE gating (router)
DECLARE_PLATFORM(moe_gate, ENGINE_CPU);

// MoE expert FFN
DECLARE_PLATFORM(moe_expert_ffn, ENGINE_CPU);

// Sparse matrix multiplication
DECLARE_PLATFORM(sparse_mul_mat, ENGINE_CPU);

// Load balancing loss for MoE
DECLARE_PLATFORM(load_balance_loss, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// Mixture of Experts Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(moe_gate, ENGINE_CUDA);
DECLARE_PLATFORM(moe_expert_ffn, ENGINE_CUDA);
DECLARE_PLATFORM(sparse_mul_mat, ENGINE_CUDA);
DECLARE_PLATFORM(load_balance_loss, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////
// KV Cache Operations - CPU
//////////////////////////////////////////////////////////////////////////

// Update KV cache
DECLARE_PLATFORM(kv_cache_update, ENGINE_CPU);

// KV cache attention
DECLARE_PLATFORM(kv_cache_attention, ENGINE_CPU);

// Paged attention
DECLARE_PLATFORM(paged_attention, ENGINE_CPU);

// Sliding window attention
DECLARE_PLATFORM(sliding_window_attention, ENGINE_CPU);

// Grouped query attention
DECLARE_PLATFORM(gqa_attention, ENGINE_CPU);

//////////////////////////////////////////////////////////////////////////
// KV Cache Operations - CUDA
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(kv_cache_update, ENGINE_CUDA);
DECLARE_PLATFORM(kv_cache_attention, ENGINE_CUDA);
DECLARE_PLATFORM(paged_attention, ENGINE_CUDA);
DECLARE_PLATFORM(sliding_window_attention, ENGINE_CUDA);
DECLARE_PLATFORM(gqa_attention, ENGINE_CUDA);

}  // namespace platforms
}  // namespace ops

namespace llamacppUtils {

#if HAVE_LLAMACPP

//////////////////////////////////////////////////////////////////////////
// CUDA Backend Support
//////////////////////////////////////////////////////////////////////////

#if defined(__CUDACC__) || defined(GGML_USE_CUDA)

/**
 * Check if CUDA backend is available
 */
bool hasCudaBackend();

/**
 * Get the number of available CUDA devices
 */
int getCudaDeviceCount();

/**
 * Set the current CUDA device for GGML operations
 */
void setCudaDevice(int deviceId);

/**
 * Get GGML CUDA backend handle
 */
struct ggml_backend* getCudaBackend(int deviceId = 0);

/**
 * Create a GGML tensor on CUDA device
 */
struct ggml_tensor* createGgmlTensorCuda(struct ggml_context* ctx, const NDArray* array,
                                          struct ggml_backend* backend, const char* name = nullptr);

/**
 * Copy GGML tensor data from CUDA to host NDArray
 */
void copyGgmlCudaToNDArray(const struct ggml_tensor* tensor, NDArray* array,
                            struct ggml_backend* backend);

/**
 * Execute a GGML computation graph on CUDA
 */
void executeGgmlGraphCuda(struct ggml_context* ctx, struct ggml_cgraph* graph,
                          struct ggml_backend* backend);

/**
 * GGML CUDA context wrapper for RAII
 */
class GgmlCudaContextGuard {
public:
    explicit GgmlCudaContextGuard(size_t memSize = 256 * 1024 * 1024, int deviceId = 0);
    ~GgmlCudaContextGuard();

    struct ggml_context* get() { return _ctx; }
    struct ggml_backend* getBackend() { return _backend; }
    operator struct ggml_context*() { return _ctx; }

private:
    struct ggml_context* _ctx;
    struct ggml_backend* _backend;
    struct ggml_backend_buffer* _buffer;
    int _deviceId;
};

#endif  // CUDA support

//////////////////////////////////////////////////////////////////////////
// CPU Backend Support
//////////////////////////////////////////////////////////////////////////

/**
 * Convert NDArray data type to GGML data type
 */
ggml_type toGgmlType(DataType dt);

/**
 * Convert GGML data type to NDArray data type
 */
DataType fromGgmlType(ggml_type gt);

/**
 * Check if a data type is supported by GGML
 */
bool isSupportedType(DataType dt);

/**
 * Check if quantized operations are available
 */
bool hasQuantizedSupport();

/**
 * Get the GGML context for operations
 * Uses thread-local storage for thread safety
 */
struct ggml_context* getGgmlContext();

/**
 * Release GGML context resources
 */
void releaseGgmlContext(struct ggml_context* ctx);

/**
 * Create a GGML tensor from an NDArray
 * @param ctx GGML context
 * @param array Source NDArray
 * @param name Optional tensor name
 * @return GGML tensor wrapping the NDArray data
 */
struct ggml_tensor* createGgmlTensor(struct ggml_context* ctx, const NDArray* array, const char* name = nullptr);

/**
 * Copy GGML tensor data back to NDArray
 * @param tensor Source GGML tensor
 * @param array Destination NDArray
 */
void copyGgmlToNDArray(const struct ggml_tensor* tensor, NDArray* array);

/**
 * Execute a GGML computation graph
 * @param ctx GGML context
 * @param graph GGML computation graph
 * @param numThreads Number of threads to use
 */
void executeGgmlGraph(struct ggml_context* ctx, struct ggml_cgraph* graph, int numThreads = 0);

/**
 * Get available GGML backends
 * @return Bitmask of available backends
 */
int getAvailableBackends();

/**
 * Backend flags
 */
enum GgmlBackend {
    GGML_BACKEND_CPU = 1 << 0,
    GGML_BACKEND_CUDA = 1 << 1,
    GGML_BACKEND_METAL = 1 << 2,
    GGML_BACKEND_VULKAN = 1 << 3,
    GGML_BACKEND_SYCL = 1 << 4,
    GGML_BACKEND_OPENCL = 1 << 5
};

/**
 * Get the preferred backend based on availability
 * @return Best available backend
 */
GgmlBackend getPreferredBackend();

/**
 * GGML context wrapper for RAII
 */
class GgmlContextGuard {
public:
    explicit GgmlContextGuard(size_t memSize = 256 * 1024 * 1024);
    ~GgmlContextGuard();

    struct ggml_context* get() { return _ctx; }
    operator struct ggml_context*() { return _ctx; }

private:
    struct ggml_context* _ctx;
    std::vector<uint8_t> _buffer;
};

//////////////////////////////////////////////////////////////////////////
// Model Parallelism Support
//////////////////////////////////////////////////////////////////////////

/**
 * Model parallelism configuration for LlamaCpp operations
 */
struct LlamaParallelConfig {
    int tensorParallelSize = 1;      // Number of devices for tensor parallelism
    int pipelineParallelSize = 1;    // Number of pipeline stages
    int numLayers = 0;               // Total number of layers (0 = auto-detect)
    std::vector<int> deviceIds;      // Device IDs to use
    bool enableP2P = true;           // Enable P2P transfers
    bool enableAsyncTransfer = true; // Enable async data movement
    size_t kvCachePerDevice = 0;     // KV cache size per device (0 = auto)
    float gpuMemoryFraction = 0.9f;  // Fraction of GPU memory to use
};

/**
 * Check if model parallelism is supported
 */
bool supportsModelParallel();

/**
 * Get the number of available devices for model parallelism
 */
int getModelParallelDeviceCount();

/**
 * Initialize model parallelism with configuration
 * @param config Parallelism configuration
 * @return true if initialization succeeded
 */
bool initModelParallel(const LlamaParallelConfig& config);

/**
 * Shutdown model parallelism and release resources
 */
void shutdownModelParallel();

/**
 * Get current model parallel configuration
 */
LlamaParallelConfig getModelParallelConfig();

/**
 * Get the device assignment for a specific layer
 * @param layerIndex Layer index
 * @return Device ID for this layer
 */
int getLayerDevice(int layerIndex);

/**
 * Partition a weight tensor for tensor parallelism
 * @param weight Weight tensor to partition
 * @param outputDim Dimension to split (typically output features)
 * @param deviceId Target device for this partition
 * @param partitionIndex Index of this partition
 * @param totalPartitions Total number of partitions
 * @return Partitioned weight tensor
 */
NDArray* partitionWeight(
    const NDArray* weight,
    int outputDim,
    int deviceId,
    int partitionIndex,
    int totalPartitions
);

/**
 * Gather output tensors from tensor parallel devices
 * @param partitions Output partitions from each device
 * @param dim Dimension to gather along
 * @param targetDevice Device for the gathered result
 * @return Gathered tensor
 */
NDArray* gatherTensorParallel(
    const std::vector<NDArray*>& partitions,
    int dim,
    int targetDevice = 0
);

/**
 * All-reduce gradients across tensor parallel devices
 * @param tensor Gradient tensor
 * @param deviceIds Devices participating in all-reduce
 */
void allReduceTensorParallel(NDArray* tensor, const std::vector<int>& deviceIds);

/**
 * Transfer activations between pipeline stages
 * @param activations Activation tensor
 * @param srcDevice Source device
 * @param dstDevice Destination device
 * @param async Whether to do async transfer
 */
void transferActivations(
    NDArray* activations,
    int srcDevice,
    int dstDevice,
    bool async = true
);

/**
 * Synchronize all devices participating in model parallelism
 */
void syncModelParallel();

/**
 * Get memory usage statistics for model parallel execution
 * @param deviceId Device to query
 * @return Memory usage in bytes
 */
size_t getModelParallelMemoryUsage(int deviceId);

//////////////////////////////////////////////////////////////////////////
// Backend Capability Tracking
//////////////////////////////////////////////////////////////////////////

/**
 * Detailed capability information for a backend
 */
struct BackendCapability {
    GgmlBackend type;
    std::string name;
    bool available;
    bool supportsQuantization;
    bool supportsFlashAttention;
    bool supportsModelParallel;
    bool supportsP2P;
    bool supportsAsyncCopy;
    bool supportsUnifiedMemory;
    std::vector<ggml_type> supportedQuantTypes;
    float relativeMatmulSpeed;
    size_t maxMemoryBytes;
    int computeCapability;
};

/**
 * Get list of all available backends
 */
std::vector<GgmlBackend> getAvailableBackendsList();

/**
 * Check if a specific backend is available
 */
bool isBackendAvailable(GgmlBackend backend);

/**
 * Get human-readable name for a backend
 */
const char* getBackendName(GgmlBackend backend);

/**
 * Check if a backend supports model parallelism
 */
bool backendSupportsModelParallel(GgmlBackend backend);

/**
 * Check if a backend supports flash attention
 */
bool backendSupportsFlashAttention(GgmlBackend backend);

/**
 * Check if a backend supports quantization
 */
bool backendSupportsQuantization(GgmlBackend backend);

/**
 * Check if a backend supports P2P transfers
 */
bool backendSupportsP2P(GgmlBackend backend);

/**
 * Get the full capability struct for a backend
 */
BackendCapability getBackendCapability(GgmlBackend backend);

/**
 * Select best backend based on requirements and data locality
 * @param input Input array to check locality
 * @param requiresQuantization Whether operation needs quantization
 * @param requiresFlashAttention Whether operation needs flash attention
 * @param requiresModelParallel Whether operation needs model parallelism
 * @return Best available backend meeting requirements
 */
GgmlBackend selectBestBackend(
    const NDArray* input = nullptr,
    bool requiresQuantization = false,
    bool requiresFlashAttention = false,
    bool requiresModelParallel = false
);

/**
 * Discover and initialize all available backends
 * Call once at startup
 */
void discoverBackends();

/**
 * Print capabilities of all available backends
 * Useful for debugging and diagnostics
 */
void printAllBackendCapabilities();

//////////////////////////////////////////////////////////////////////////
// Device Locality Management
//////////////////////////////////////////////////////////////////////////

/**
 * Represents where data is currently valid/authoritative
 */
enum class DataLocation {
    UNKNOWN = 0,
    HOST_ONLY = 1,      // Data only valid in primary buffer (CPU)
    DEVICE_ONLY = 2,    // Data only valid in special buffer (GPU)
    SYNCHRONIZED = 3,   // Data valid in both buffers
    STALE = 4           // Data needs refresh from source
};

/**
 * Execution context that tracks data locality for GGML operations
 */
struct LocalityContext {
    int targetDeviceId;
    GgmlBackend targetBackend;
    bool requiresHostData;
    bool requiresDeviceData;
    bool allowDataMovement;
    size_t estimatedMemoryUsage;

    // Statistics
    size_t hostToDeviceTransfers;
    size_t deviceToHostTransfers;
    size_t deviceToDeviceTransfers;
};

/**
 * Get the current valid location of an NDArray's data
 */
DataLocation getDataLocation(const NDArray* array);

/**
 * Get string representation of data location
 */
const char* getDataLocationString(DataLocation loc);

/**
 * Get the current thread's locality context
 */
LocalityContext& getLocalityContext();

/**
 * Set up locality context for an operation
 */
void setupLocalityContext(int deviceId, GgmlBackend backend, size_t memoryNeeded);

/**
 * Ensure NDArray data is available on host (primary buffer)
 * @param array NDArray to synchronize
 * @param forWrite Whether data will be modified
 * @return Pointer to host data or nullptr on failure
 */
void* ensureDataOnHost(NDArray* array, bool forWrite);

/**
 * Ensure NDArray data is available on device (special buffer)
 * @param array NDArray to synchronize
 * @param deviceId Target device
 * @param forWrite Whether data will be modified
 * @return Pointer to device data or nullptr on failure
 */
void* ensureDataOnDevice(NDArray* array, int deviceId, bool forWrite);

/**
 * Prepare an NDArray for a GGML operation based on target backend
 * Handles data synchronization to ensure data is in the right place
 * @param array Array to prepare
 * @param backend Target GGML backend
 * @param isInput Whether array is input (read-only) or output
 * @return true on success
 */
bool prepareArrayForGgml(NDArray* array, GgmlBackend backend, bool isInput);

/**
 * Finalize array after GGML operation (mark output location)
 */
void finalizeArrayAfterGgml(NDArray* array, GgmlBackend backend);

/**
 * Transfer data between devices
 * Handles the primary/special buffer synchronization
 * @param array Array to transfer
 * @param sourceDevice Source device ID
 * @param targetDevice Target device ID
 * @return true on success
 */
bool transferDataBetweenDevices(NDArray* array, int sourceDevice, int targetDevice);

/**
 * Check if P2P transfer is possible between devices
 */
bool canDoP2PTransfer(int sourceDevice, int targetDevice);

/**
 * Create a GGML tensor that wraps NDArray data with proper locality
 * This is the key integration point with ND4J's buffer system
 * @param ctx GGML context
 * @param array Source NDArray
 * @param backend Target backend
 * @param name Optional tensor name
 * @return GGML tensor or nullptr on failure
 */
struct ggml_tensor* createGgmlTensorWithLocality(
    struct ggml_context* ctx,
    NDArray* array,
    GgmlBackend backend,
    const char* name = nullptr
);

/**
 * Copy GGML tensor result back to NDArray with proper locality handling
 * @param tensor Source GGML tensor
 * @param array Destination NDArray
 * @param backend Backend that produced the result
 */
void copyGgmlResultWithLocality(
    const struct ggml_tensor* tensor,
    NDArray* array,
    GgmlBackend backend
);

/**
 * Get locality transfer statistics for current thread
 * @param h2d Host to device transfer count (output)
 * @param d2h Device to host transfer count (output)
 * @param d2d Device to device transfer count (output)
 */
void getLocalityStatistics(size_t* h2d, size_t* d2h, size_t* d2d);

/**
 * Reset locality statistics
 */
void resetLocalityStatistics();

/**
 * Print locality information for an array
 * Useful for debugging data movement issues
 */
void printArrayLocalityInfo(const NDArray* array, const char* name = nullptr);

/**
 * Validate that all inputs are on the expected device for an operation
 * @param inputs Input arrays to check
 * @param expectedDevice Expected device ID
 * @return true if all inputs are on expected device
 */
bool validateInputLocality(const std::vector<const NDArray*>& inputs, int expectedDevice);

//////////////////////////////////////////////////////////////////////////
// Per-Device Backend State
//////////////////////////////////////////////////////////////////////////

/**
 * State tracking for a backend on a specific device
 */
struct DeviceBackendState {
    GgmlBackend backend;
    int deviceId;
    bool initialized;
    size_t memoryAllocated;
    size_t memoryUsed;
    size_t peakMemoryUsed;
    int activeComputeGraphs;
    bool busy;
};

/**
 * Get or create backend state for a device
 */
DeviceBackendState& getDeviceBackendState(GgmlBackend backend, int deviceId);

/**
 * Initialize a backend for a specific device
 */
bool initializeBackendForDevice(GgmlBackend backend, int deviceId);

/**
 * Shutdown backend on a device and release resources
 */
void shutdownBackendForDevice(GgmlBackend backend, int deviceId);

/**
 * Get memory usage for a backend on a device
 */
size_t getBackendMemoryUsage(GgmlBackend backend, int deviceId);

/**
 * Get peak memory usage for a backend on a device
 */
size_t getBackendPeakMemoryUsage(GgmlBackend backend, int deviceId);

/**
 * Reset peak memory tracking for a backend
 */
void resetBackendPeakMemory(GgmlBackend backend, int deviceId);

#endif  // HAVE_LLAMACPP

}  // namespace llamacppUtils
}  // namespace sd

#endif  // DEV_TESTS_LLAMACPPUTILS_H
