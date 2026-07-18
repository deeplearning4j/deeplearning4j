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
// Metal Performance Shaders (MPS) helper utilities for Apple GPU acceleration
// Provides GPU-accelerated operations on Apple Silicon and AMD GPUs in Macs
//

#ifndef SD_MPSUTILS_H
#define SD_MPSUTILS_H

#include <array/NDArray.h>
#include <graph/Context.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <system/RequirementsHelper.h>
#include <ConstMessages.h>

#ifdef HAVE_MPS
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

// IS_USE_MPS_MSG is declared as extern const char* in ConstMessages.h and
// defined in ConstMessages.cpp.  The #include <ConstMessages.h> above makes
// it available to all translation units that include this header.

#include <system/BackendNamespace.h>

namespace sd {
namespace ops {
namespace platforms {
// No file-level SD_NS wrap: DECLARE_PLATFORM opens it per declaration
// (platform_boilerplate.h); wrapping the list again nests SD_NS in itself.

/**
 * Platform helper declarations for Apple Metal Performance Shaders (MPS)
 * These operations run on the GPU using Apple's MPS framework
 */

// ============================================================================
// BLAS Operations (via MPSMatrixMultiplication)
// ============================================================================

// Matrix multiplication on GPU
DECLARE_PLATFORM(matmul, ENGINE_CPU);  // Note: MPS uses ENGINE_CPU as the selection engine

// Batch matrix multiplication
DECLARE_PLATFORM(batched_gemm, ENGINE_CPU);

// ============================================================================
// Convolution Operations (via MPSCNNConvolution)
// ============================================================================

// 2D Convolution
DECLARE_PLATFORM(conv2d, ENGINE_CPU);
DECLARE_PLATFORM(conv2d_bp, ENGINE_CPU);

// Depthwise separable convolution
DECLARE_PLATFORM(depthwise_conv2d, ENGINE_CPU);
DECLARE_PLATFORM(depthwise_conv2d_bp, ENGINE_CPU);

// ============================================================================
// Pooling Operations (via MPSCNNPooling)
// ============================================================================

// Max pooling 2D
DECLARE_PLATFORM(maxpool2d, ENGINE_CPU);
DECLARE_PLATFORM(maxpool2d_bp, ENGINE_CPU);

// Average pooling 2D
DECLARE_PLATFORM(avgpool2d, ENGINE_CPU);
DECLARE_PLATFORM(avgpool2d_bp, ENGINE_CPU);

// ============================================================================
// Normalization Operations
// ============================================================================

// Batch normalization (via MPSCNNBatchNormalization)
DECLARE_PLATFORM(batchnorm, ENGINE_CPU);
DECLARE_PLATFORM(batchnorm_bp, ENGINE_CPU);

// Instance normalization
DECLARE_PLATFORM(instance_norm, ENGINE_CPU);

// Layer normalization
DECLARE_PLATFORM(layer_norm, ENGINE_CPU);

// ============================================================================
// Activation Functions (via MPSCNNNeuron)
// ============================================================================

// ReLU activation
DECLARE_PLATFORM(relu, ENGINE_CPU);

// Leaky ReLU
DECLARE_PLATFORM(leaky_relu, ENGINE_CPU);

// ELU activation
DECLARE_PLATFORM(elu, ENGINE_CPU);

// GELU activation
DECLARE_PLATFORM(gelu, ENGINE_CPU);

// Softmax
DECLARE_PLATFORM(softmax, ENGINE_CPU);

// Sigmoid
DECLARE_PLATFORM(sigmoid, ENGINE_CPU);

// Tanh
DECLARE_PLATFORM(tanh, ENGINE_CPU);

// SiLU/Swish
DECLARE_PLATFORM(silu, ENGINE_CPU);

// ============================================================================
// Reduction Operations
// ============================================================================

// Sum reduction
DECLARE_PLATFORM(reduce_sum, ENGINE_CPU);

// Mean reduction
DECLARE_PLATFORM(reduce_mean, ENGINE_CPU);

// Max reduction
DECLARE_PLATFORM(reduce_max, ENGINE_CPU);

// Min reduction
DECLARE_PLATFORM(reduce_min, ENGINE_CPU);

// Product reduction
DECLARE_PLATFORM(reduce_prod, ENGINE_CPU);

// Variance reduction
DECLARE_PLATFORM(reduce_variance, ENGINE_CPU);

// Standard deviation reduction
DECLARE_PLATFORM(reduce_stdev, ENGINE_CPU);

// ============================================================================
// Element-wise Operations
// ============================================================================

// Element-wise addition
DECLARE_PLATFORM(add, ENGINE_CPU);

// Element-wise subtraction
DECLARE_PLATFORM(subtract, ENGINE_CPU);

// Element-wise multiplication
DECLARE_PLATFORM(multiply, ENGINE_CPU);

// Element-wise division
DECLARE_PLATFORM(divide, ENGINE_CPU);

// Element-wise square root
DECLARE_PLATFORM(sqrt, ENGINE_CPU);

// Element-wise exponential
DECLARE_PLATFORM(exp, ENGINE_CPU);

// Element-wise natural logarithm
DECLARE_PLATFORM(log, ENGINE_CPU);

// Element-wise power
DECLARE_PLATFORM(pow, ENGINE_CPU);

// Element-wise absolute value
DECLARE_PLATFORM(abs, ENGINE_CPU);

// Element-wise negative
DECLARE_PLATFORM(neg, ENGINE_CPU);

// ============================================================================
// Image Operations (via MPSImageConvolution, etc.)
// ============================================================================

// Image resize/resample
DECLARE_PLATFORM(resize_bilinear, ENGINE_CPU);
DECLARE_PLATFORM(resize_nearest, ENGINE_CPU);

// Crop and resize
DECLARE_PLATFORM(crop_and_resize, ENGINE_CPU);

// ============================================================================
// RNN/LSTM/GRU Operations
// ============================================================================

// Simple RNN cell
DECLARE_PLATFORM(simple_rnn, ENGINE_CPU);

// LSTM cell
DECLARE_PLATFORM(lstmCell, ENGINE_CPU);

// GRU cell
DECLARE_PLATFORM(gruCell, ENGINE_CPU);

// Bidirectional RNN
DECLARE_PLATFORM(static_bidirectional_rnn, ENGINE_CPU);

// ============================================================================
// Transform Operations
// ============================================================================

// Reshape
DECLARE_PLATFORM(reshape, ENGINE_CPU);

// Flatten
DECLARE_PLATFORM(flatten, ENGINE_CPU);

// Squeeze/Unsqueeze
DECLARE_PLATFORM(squeeze, ENGINE_CPU);
DECLARE_PLATFORM(expand_dims, ENGINE_CPU);

// Permute/Transpose
DECLARE_PLATFORM(permute, ENGINE_CPU);
DECLARE_PLATFORM(transpose, ENGINE_CPU);

// Split
DECLARE_PLATFORM(split, ENGINE_CPU);

// Space/Depth transformations
DECLARE_PLATFORM(space_to_depth, ENGINE_CPU);
DECLARE_PLATFORM(depth_to_space, ENGINE_CPU);
DECLARE_PLATFORM(batch_to_space_nd, ENGINE_CPU);
DECLARE_PLATFORM(space_to_batch_nd, ENGINE_CPU);

// ============================================================================
// Embedding Operations
// ============================================================================

// Embedding lookup
DECLARE_PLATFORM(embedding_lookup, ENGINE_CPU);

// One-hot encoding
DECLARE_PLATFORM(onehot, ENGINE_CPU);

// Segment operations
DECLARE_PLATFORM(segment_sum, ENGINE_CPU);
DECLARE_PLATFORM(segment_mean, ENGINE_CPU);
DECLARE_PLATFORM(segment_max, ENGINE_CPU);
DECLARE_PLATFORM(segment_min, ENGINE_CPU);
DECLARE_PLATFORM(segment_prod, ENGINE_CPU);
DECLARE_PLATFORM(unsorted_segment_sum, ENGINE_CPU);

// ============================================================================
// Attention Operations
// ============================================================================

// Scaled dot-product attention
DECLARE_PLATFORM(dot_product_attention, ENGINE_CPU);

// Multi-head attention
DECLARE_PLATFORM(multi_head_dot_product_attention, ENGINE_CPU);

// Additive attention (Bahdanau)
DECLARE_PLATFORM(additive_attention, ENGINE_CPU);

// Self-attention
DECLARE_PLATFORM(self_attention, ENGINE_CPU);

// ============================================================================
// Sorting and Unique Operations
// ============================================================================

// Sort and argsort
DECLARE_PLATFORM(sort, ENGINE_CPU);
DECLARE_PLATFORM(argsort, ENGINE_CPU);

// Top-K operations
DECLARE_PLATFORM(top_k, ENGINE_CPU);
DECLARE_PLATFORM(in_top_k, ENGINE_CPU);

// Unique operations
DECLARE_PLATFORM(unique, ENGINE_CPU);
DECLARE_PLATFORM(unique_with_counts, ENGINE_CPU);

// Argmax/Argmin
DECLARE_PLATFORM(argmax, ENGINE_CPU);
DECLARE_PLATFORM(argmin, ENGINE_CPU);

// Histogram operations
DECLARE_PLATFORM(histogram, ENGINE_CPU);
DECLARE_PLATFORM(bincount, ENGINE_CPU);

// ============================================================================
// Comparison Operations (from mps_comparison.mm)
// ============================================================================

DECLARE_PLATFORM(greater, ENGINE_CPU);
DECLARE_PLATFORM(greater_equal, ENGINE_CPU);
DECLARE_PLATFORM(less, ENGINE_CPU);
DECLARE_PLATFORM(less_equal, ENGINE_CPU);
DECLARE_PLATFORM(equals, ENGINE_CPU);
DECLARE_PLATFORM(not_equals, ENGINE_CPU);
DECLARE_PLATFORM(maximum, ENGINE_CPU);
DECLARE_PLATFORM(minimum, ENGINE_CPU);
DECLARE_PLATFORM(where_np, ENGINE_CPU);

// ============================================================================
// Math Operations (from mps_math.mm)
// ============================================================================

DECLARE_PLATFORM(sin, ENGINE_CPU);
DECLARE_PLATFORM(cos, ENGINE_CPU);
DECLARE_PLATFORM(tan, ENGINE_CPU);
DECLARE_PLATFORM(asin, ENGINE_CPU);
DECLARE_PLATFORM(acos, ENGINE_CPU);
DECLARE_PLATFORM(atan, ENGINE_CPU);
DECLARE_PLATFORM(atan2, ENGINE_CPU);
DECLARE_PLATFORM(sinh, ENGINE_CPU);
DECLARE_PLATFORM(cosh, ENGINE_CPU);
DECLARE_PLATFORM(asinh, ENGINE_CPU);
DECLARE_PLATFORM(acosh, ENGINE_CPU);
DECLARE_PLATFORM(atanh, ENGINE_CPU);
DECLARE_PLATFORM(Floor, ENGINE_CPU);
DECLARE_PLATFORM(Ceil, ENGINE_CPU);
DECLARE_PLATFORM(Round, ENGINE_CPU);
DECLARE_PLATFORM(Sign, ENGINE_CPU);
DECLARE_PLATFORM(clip_by_value, ENGINE_CPU);
DECLARE_PLATFORM(reciprocal, ENGINE_CPU);
DECLARE_PLATFORM(square, ENGINE_CPU);
DECLARE_PLATFORM(cube, ENGINE_CPU);
DECLARE_PLATFORM(rsqrt, ENGINE_CPU);
DECLARE_PLATFORM(log1p, ENGINE_CPU);
DECLARE_PLATFORM(expm1, ENGINE_CPU);
DECLARE_PLATFORM(erf, ENGINE_CPU);
DECLARE_PLATFORM(erfc, ENGINE_CPU);

// ============================================================================
// Matrix Operations (from mps_matrix.mm)
// ============================================================================

DECLARE_PLATFORM(concat, ENGINE_CPU);
DECLARE_PLATFORM(stack, ENGINE_CPU);
DECLARE_PLATFORM(unstack, ENGINE_CPU);
DECLARE_PLATFORM(tile, ENGINE_CPU);
DECLARE_PLATFORM(repeat, ENGINE_CPU);
DECLARE_PLATFORM(reverse_sequence, ENGINE_CPU);
DECLARE_PLATFORM(pad, ENGINE_CPU);
DECLARE_PLATFORM(slice, ENGINE_CPU);
DECLARE_PLATFORM(strided_slice, ENGINE_CPU);
DECLARE_PLATFORM(scatter_update, ENGINE_CPU);
DECLARE_PLATFORM(gather, ENGINE_CPU);
DECLARE_PLATFORM(gather_nd, ENGINE_CPU);
DECLARE_PLATFORM(reverse, ENGINE_CPU);

// ============================================================================
// Extended Activation Operations (from mps_activations_ext.mm)
// ============================================================================

DECLARE_PLATFORM(hard_sigmoid, ENGINE_CPU);
DECLARE_PLATFORM(hardswish, ENGINE_CPU);
DECLARE_PLATFORM(mish, ENGINE_CPU);
DECLARE_PLATFORM(softplus, ENGINE_CPU);
DECLARE_PLATFORM(softsign, ENGINE_CPU);
DECLARE_PLATFORM(prelu, ENGINE_CPU);
DECLARE_PLATFORM(selu, ENGINE_CPU);
DECLARE_PLATFORM(celu, ENGINE_CPU);
DECLARE_PLATFORM(relu6, ENGINE_CPU);
DECLARE_PLATFORM(thresholdedrelu, ENGINE_CPU);
DECLARE_PLATFORM(log_softmax, ENGINE_CPU);

// ============================================================================
// Loss Operations (from mps_loss.mm)
// ============================================================================

DECLARE_PLATFORM(mean_sqerr_loss, ENGINE_CPU);
DECLARE_PLATFORM(mean_absolute_error, ENGINE_CPU);
DECLARE_PLATFORM(huber_loss, ENGINE_CPU);
DECLARE_PLATFORM(sigm_cross_entropy_loss, ENGINE_CPU);
DECLARE_PLATFORM(softmax_cross_entropy_loss, ENGINE_CPU);
DECLARE_PLATFORM(sparse_softmax_cross_entropy_loss_with_logits, ENGINE_CPU);
DECLARE_PLATFORM(cosine_distance_loss, ENGINE_CPU);
DECLARE_PLATFORM(hinge_loss, ENGINE_CPU);
DECLARE_PLATFORM(log_loss, ENGINE_CPU);

}  // namespace platforms

SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
namespace mpsUtils {

// ============================================================================
// These declarations are available regardless of HAVE_MPS so that
// PLATFORM_CHECK implementations compile on all platforms.
// ============================================================================

/**
 * Check if MPS helper is available at runtime.
 * Returns false when HAVE_MPS is not defined.
 */
bool hasMPSSupport();

/**
 * Returns true when the NDArray is contiguous in C order (row-major).
 * Available unconditionally so PLATFORM_CHECK can call it on any platform.
 */
bool isContiguous(const sd::NDArray& arr);

/**
 * Singleton device manager for MPS / Metal.
 *
 * On non-Apple platforms (or when HAVE_MPS is not defined) all methods
 * are no-ops / return nullptr / false so the rest of the code compiles
 * unconditionally.
 */
class MPSDeviceManager {
public:
    static MPSDeviceManager& getInstance();

    bool initialize();
    void shutdown();

    /** Returns true only when a Metal device was successfully acquired. */
    bool isAvailable() const { return _initialized && _available; }

#ifdef HAVE_MPS
    id<MTLDevice>        getDevice()       const { return _device; }
    id<MTLCommandQueue>  getCommandQueue() const { return _commandQueue; }
    id<MTLCommandBuffer> createCommandBuffer();
    std::string          getDeviceName()   const;
    size_t               getMaxMemory()    const;
    bool                 supportsFamily(MTLGPUFamily family) const;
#else
    void* getDevice()       const { return nullptr; }
    void* getCommandQueue() const { return nullptr; }
#endif

private:
    MPSDeviceManager();
    ~MPSDeviceManager();
    MPSDeviceManager(const MPSDeviceManager&) = delete;
    MPSDeviceManager& operator=(const MPSDeviceManager&) = delete;

    bool _initialized = false;
    bool _available   = false;

#ifdef HAVE_MPS
    id<MTLDevice>       _device       = nil;
    id<MTLCommandQueue> _commandQueue = nil;
#endif
};

// ============================================================================
// MPS-only helpers (compiled only when Metal is present)
// ============================================================================

#ifdef HAVE_MPS

/** Returns true if the data type has a native MPS representation. */
bool isMPSSupported(sd::DataType dtype);

/** Returns true if the array is contiguous and has a MPS-supported dtype. */
bool isMPSFriendly(const sd::NDArray& arr);

/** Map a libnd4j DataType to the corresponding MPSDataType. */
MPSDataType getMPSDataType(sd::DataType dtype);

/** Create an MPSMatrix wrapping the data in @p arr. */
MPSMatrix* createMPSMatrix(const sd::NDArray* arr, id<MTLDevice> device);

/** Create an MPSImage from a 4-D NDArray (NCHW). */
MPSImage* createMPSImage(const sd::NDArray* arr, id<MTLDevice> device);

/** Synchronously copy an MPSMatrix result back to @p arr. */
void copyMPSMatrixToNDArray(MPSMatrix* matrix, sd::NDArray* arr);

/** Synchronously copy an MPSImage result back to @p arr. */
void copyMPSImageToNDArray(MPSImage* image, sd::NDArray* arr);

/** Wait for all pending GPU work to complete. */
void synchronize();

/** RAII wrapper for an MTLCommandBuffer. */
class MPSCommandBufferGuard {
public:
    MPSCommandBufferGuard();
    ~MPSCommandBufferGuard();

    id<MTLCommandBuffer> get()  const { return _commandBuffer; }
    operator id<MTLCommandBuffer>() const { return _commandBuffer; }

    void commitAndWait();
    void commit();

private:
    id<MTLCommandBuffer> _commandBuffer = nil;
    bool _committed = false;
};

/** Fill @p reqs with the standard MPS preconditions. */
void checkMPSRequirements(sd::Requirements& reqs, sd::graph::Context& block,
                           const sd::NDArray* input  = nullptr,
                           const sd::NDArray* output = nullptr);

#endif  // HAVE_MPS

}  // namespace mpsUtils
SD_BACKEND_OPS_INLINE_NAMESPACE_END
}  // namespace ops
}  // namespace sd

#endif  // SD_MPSUTILS_H
