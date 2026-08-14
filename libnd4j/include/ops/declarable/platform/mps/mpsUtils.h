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
#import <Foundation/Foundation.h>
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
DECLARE_PLATFORM(matmul, ENGINE_MPS);  // Note: MPS uses ENGINE_MPS as the selection engine

// Batch matrix multiplication
DECLARE_PLATFORM(batched_gemm, ENGINE_MPS);

// ============================================================================
// Convolution Operations (via MPSCNNConvolution)
// ============================================================================

// 2D Convolution
DECLARE_PLATFORM(conv2d, ENGINE_MPS);
DECLARE_PLATFORM(conv2d_bp, ENGINE_MPS);

// Depthwise separable convolution
DECLARE_PLATFORM(depthwise_conv2d, ENGINE_MPS);
DECLARE_PLATFORM(depthwise_conv2d_bp, ENGINE_MPS);

// ============================================================================
// Pooling Operations (via MPSCNNPooling)
// ============================================================================

// Max pooling 2D
DECLARE_PLATFORM(maxpool2d, ENGINE_MPS);
DECLARE_PLATFORM(maxpool2d_bp, ENGINE_MPS);

// Average pooling 2D
DECLARE_PLATFORM(avgpool2d, ENGINE_MPS);
DECLARE_PLATFORM(avgpool2d_bp, ENGINE_MPS);

// ============================================================================
// Normalization Operations
// ============================================================================

// Batch normalization (via MPSCNNBatchNormalization)
DECLARE_PLATFORM(batchnorm, ENGINE_MPS);
DECLARE_PLATFORM(batchnorm_bp, ENGINE_MPS);

// Instance normalization
DECLARE_PLATFORM(instance_norm, ENGINE_MPS);

// Layer normalization
DECLARE_PLATFORM(layer_norm, ENGINE_MPS);

// ============================================================================
// Activation Functions (via MPSCNNNeuron)
// ============================================================================

// ReLU activation
DECLARE_PLATFORM(relu, ENGINE_MPS);

// Leaky ReLU
DECLARE_PLATFORM(leaky_relu, ENGINE_MPS);

// ELU activation
DECLARE_PLATFORM(elu, ENGINE_MPS);

// GELU activation
DECLARE_PLATFORM(gelu, ENGINE_MPS);

// Softmax
DECLARE_PLATFORM(softmax, ENGINE_MPS);

// Sigmoid
DECLARE_PLATFORM(sigmoid, ENGINE_MPS);

// Tanh
DECLARE_PLATFORM(tanh, ENGINE_MPS);

// SiLU/Swish
DECLARE_PLATFORM(silu, ENGINE_MPS);

// ============================================================================
// Reduction Operations
// ============================================================================

// Sum reduction
DECLARE_PLATFORM(reduce_sum, ENGINE_MPS);

// Mean reduction
DECLARE_PLATFORM(reduce_mean, ENGINE_MPS);

// Max reduction
DECLARE_PLATFORM(reduce_max, ENGINE_MPS);

// Min reduction
DECLARE_PLATFORM(reduce_min, ENGINE_MPS);

// Product reduction
DECLARE_PLATFORM(reduce_prod, ENGINE_MPS);

// Variance reduction
DECLARE_PLATFORM(reduce_variance, ENGINE_MPS);

// Standard deviation reduction
DECLARE_PLATFORM(reduce_stdev, ENGINE_MPS);

// ============================================================================
// Element-wise Operations
// ============================================================================

// Element-wise addition
DECLARE_PLATFORM(add, ENGINE_MPS);

// Element-wise subtraction
DECLARE_PLATFORM(subtract, ENGINE_MPS);

// Element-wise multiplication
DECLARE_PLATFORM(multiply, ENGINE_MPS);

// Element-wise division
DECLARE_PLATFORM(divide, ENGINE_MPS);

// Element-wise square root
DECLARE_PLATFORM(sqrt, ENGINE_MPS);

// Element-wise exponential
DECLARE_PLATFORM(exp, ENGINE_MPS);

// Element-wise natural logarithm
DECLARE_PLATFORM(log, ENGINE_MPS);

// Element-wise power
DECLARE_PLATFORM(pow, ENGINE_MPS);

// Element-wise absolute value
DECLARE_PLATFORM(abs, ENGINE_MPS);

// Element-wise negative
DECLARE_PLATFORM(neg, ENGINE_MPS);

// ============================================================================
// Image Operations (via MPSImageConvolution, etc.)
// ============================================================================

// Image resize/resample
DECLARE_PLATFORM(resize_bilinear, ENGINE_MPS);
DECLARE_PLATFORM(resize_nearest, ENGINE_MPS);

// Crop and resize
DECLARE_PLATFORM(crop_and_resize, ENGINE_MPS);

// ============================================================================
// RNN/LSTM/GRU Operations
// ============================================================================

// Simple RNN cell
DECLARE_PLATFORM(simple_rnn, ENGINE_MPS);

// LSTM cell
DECLARE_PLATFORM(lstmCell, ENGINE_MPS);

// GRU cell
DECLARE_PLATFORM(gruCell, ENGINE_MPS);

// Bidirectional RNN
DECLARE_PLATFORM(static_bidirectional_rnn, ENGINE_MPS);

// ============================================================================
// Transform Operations
// ============================================================================

// Reshape
DECLARE_PLATFORM(reshape, ENGINE_MPS);

// Flatten
DECLARE_PLATFORM(flatten, ENGINE_MPS);

// Squeeze/Unsqueeze
DECLARE_PLATFORM(squeeze, ENGINE_MPS);
DECLARE_PLATFORM(expand_dims, ENGINE_MPS);

// Permute/Transpose
DECLARE_PLATFORM(permute, ENGINE_MPS);
DECLARE_PLATFORM(transpose, ENGINE_MPS);

// Split
DECLARE_PLATFORM(split, ENGINE_MPS);

// Space/Depth transformations
DECLARE_PLATFORM(space_to_depth, ENGINE_MPS);
DECLARE_PLATFORM(depth_to_space, ENGINE_MPS);
DECLARE_PLATFORM(batch_to_space_nd, ENGINE_MPS);
DECLARE_PLATFORM(space_to_batch_nd, ENGINE_MPS);

// ============================================================================
// Embedding Operations
// ============================================================================

// Embedding lookup
DECLARE_PLATFORM(embedding_lookup, ENGINE_MPS);

// One-hot encoding
DECLARE_PLATFORM(onehot, ENGINE_MPS);

// Segment operations
DECLARE_PLATFORM(segment_sum, ENGINE_MPS);
DECLARE_PLATFORM(segment_mean, ENGINE_MPS);
DECLARE_PLATFORM(segment_max, ENGINE_MPS);
DECLARE_PLATFORM(segment_min, ENGINE_MPS);
DECLARE_PLATFORM(segment_prod, ENGINE_MPS);
DECLARE_PLATFORM(unsorted_segment_sum, ENGINE_MPS);

// ============================================================================
// Attention Operations
// ============================================================================

// Scaled dot-product attention
DECLARE_PLATFORM(dot_product_attention, ENGINE_MPS);

// Multi-head attention
DECLARE_PLATFORM(multi_head_dot_product_attention, ENGINE_MPS);

// Additive attention (Bahdanau)
DECLARE_PLATFORM(additive_attention, ENGINE_MPS);

// Self-attention
DECLARE_PLATFORM(self_attention, ENGINE_MPS);

// ============================================================================
// Sorting and Unique Operations
// ============================================================================

// Sort and argsort
DECLARE_PLATFORM(sort, ENGINE_MPS);
DECLARE_PLATFORM(argsort, ENGINE_MPS);

// Top-K operations
DECLARE_PLATFORM(top_k, ENGINE_MPS);
DECLARE_PLATFORM(in_top_k, ENGINE_MPS);

// Unique operations
DECLARE_PLATFORM(unique, ENGINE_MPS);
DECLARE_PLATFORM(unique_with_counts, ENGINE_MPS);

// Argmax/Argmin
DECLARE_PLATFORM(argmax, ENGINE_MPS);
DECLARE_PLATFORM(argmin, ENGINE_MPS);

// Histogram operations
DECLARE_PLATFORM(histogram, ENGINE_MPS);
DECLARE_PLATFORM(bincount, ENGINE_MPS);

// ============================================================================
// Comparison Operations (from mps_comparison.mm)
// ============================================================================

DECLARE_PLATFORM(greater, ENGINE_MPS);
DECLARE_PLATFORM(greater_equal, ENGINE_MPS);
DECLARE_PLATFORM(less, ENGINE_MPS);
DECLARE_PLATFORM(less_equal, ENGINE_MPS);
DECLARE_PLATFORM(equals, ENGINE_MPS);
DECLARE_PLATFORM(not_equals, ENGINE_MPS);
DECLARE_PLATFORM(maximum, ENGINE_MPS);
DECLARE_PLATFORM(minimum, ENGINE_MPS);
DECLARE_PLATFORM(where_np, ENGINE_MPS);

// ============================================================================
// Math Operations (from mps_math.mm)
// ============================================================================

DECLARE_PLATFORM(sin, ENGINE_MPS);
DECLARE_PLATFORM(cos, ENGINE_MPS);
DECLARE_PLATFORM(tan, ENGINE_MPS);
DECLARE_PLATFORM(asin, ENGINE_MPS);
DECLARE_PLATFORM(acos, ENGINE_MPS);
DECLARE_PLATFORM(atan, ENGINE_MPS);
DECLARE_PLATFORM(atan2, ENGINE_MPS);
DECLARE_PLATFORM(sinh, ENGINE_MPS);
DECLARE_PLATFORM(cosh, ENGINE_MPS);
DECLARE_PLATFORM(asinh, ENGINE_MPS);
DECLARE_PLATFORM(acosh, ENGINE_MPS);
DECLARE_PLATFORM(atanh, ENGINE_MPS);
DECLARE_PLATFORM(Floor, ENGINE_MPS);
DECLARE_PLATFORM(Ceil, ENGINE_MPS);
DECLARE_PLATFORM(Round, ENGINE_MPS);
DECLARE_PLATFORM(Sign, ENGINE_MPS);
DECLARE_PLATFORM(clip_by_value, ENGINE_MPS);
DECLARE_PLATFORM(reciprocal, ENGINE_MPS);
DECLARE_PLATFORM(square, ENGINE_MPS);
DECLARE_PLATFORM(cube, ENGINE_MPS);
DECLARE_PLATFORM(rsqrt, ENGINE_MPS);
DECLARE_PLATFORM(log1p, ENGINE_MPS);
DECLARE_PLATFORM(expm1, ENGINE_MPS);
DECLARE_PLATFORM(erf, ENGINE_MPS);
DECLARE_PLATFORM(erfc, ENGINE_MPS);

// ============================================================================
// Matrix Operations (from mps_matrix.mm)
// ============================================================================

DECLARE_PLATFORM(concat, ENGINE_MPS);
DECLARE_PLATFORM(stack, ENGINE_MPS);
DECLARE_PLATFORM(unstack, ENGINE_MPS);
DECLARE_PLATFORM(tile, ENGINE_MPS);
DECLARE_PLATFORM(repeat, ENGINE_MPS);
DECLARE_PLATFORM(reverse_sequence, ENGINE_MPS);
DECLARE_PLATFORM(pad, ENGINE_MPS);
DECLARE_PLATFORM(slice, ENGINE_MPS);
DECLARE_PLATFORM(strided_slice, ENGINE_MPS);
DECLARE_PLATFORM(scatter_update, ENGINE_MPS);
DECLARE_PLATFORM(gather, ENGINE_MPS);
DECLARE_PLATFORM(gather_nd, ENGINE_MPS);
DECLARE_PLATFORM(reverse, ENGINE_MPS);

// ============================================================================
// Extended Activation Operations (from mps_activations_ext.mm)
// ============================================================================

DECLARE_PLATFORM(hard_sigmoid, ENGINE_MPS);
DECLARE_PLATFORM(hardswish, ENGINE_MPS);
DECLARE_PLATFORM(mish, ENGINE_MPS);
DECLARE_PLATFORM(softplus, ENGINE_MPS);
DECLARE_PLATFORM(softsign, ENGINE_MPS);
DECLARE_PLATFORM(prelu, ENGINE_MPS);
DECLARE_PLATFORM(selu, ENGINE_MPS);
DECLARE_PLATFORM(celu, ENGINE_MPS);
DECLARE_PLATFORM(relu6, ENGINE_MPS);
DECLARE_PLATFORM(thresholdedrelu, ENGINE_MPS);
DECLARE_PLATFORM(log_softmax, ENGINE_MPS);

// ============================================================================
// Loss Operations (from mps_loss.mm)
// ============================================================================

DECLARE_PLATFORM(mean_sqerr_loss, ENGINE_MPS);
DECLARE_PLATFORM(mean_absolute_error, ENGINE_MPS);
DECLARE_PLATFORM(huber_loss, ENGINE_MPS);
DECLARE_PLATFORM(sigm_cross_entropy_loss, ENGINE_MPS);
DECLARE_PLATFORM(softmax_cross_entropy_loss, ENGINE_MPS);
DECLARE_PLATFORM(sparse_softmax_cross_entropy_loss_with_logits, ENGINE_MPS);
DECLARE_PLATFORM(cosine_distance_loss, ENGINE_MPS);
DECLARE_PLATFORM(hinge_loss, ENGINE_MPS);
DECLARE_PLATFORM(log_loss, ENGINE_MPS);

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
bool isContiguous(sd::NDArray& arr);

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
bool isMPSFriendly(sd::NDArray& arr);

/** Map a libnd4j DataType to the corresponding MPSDataType. */
MPSDataType getMPSDataType(sd::DataType dtype);

/** Create an MPSMatrix wrapping the data in @p arr. */
MPSMatrix* createMPSMatrix(sd::NDArray* arr, id<MTLDevice> device);

/** Create an MPSImage from a 4-D NDArray (NCHW). */
MPSImage* createMPSImage(sd::NDArray* arr, id<MTLDevice> device);

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
                           sd::NDArray* input  = nullptr,
                           sd::NDArray* output = nullptr);

#endif  // HAVE_MPS

}  // namespace mpsUtils
SD_BACKEND_OPS_INLINE_NAMESPACE_END
}  // namespace ops
}  // namespace sd

#endif  // SD_MPSUTILS_H
