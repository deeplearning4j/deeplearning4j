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
// Apple Accelerate framework helper utilities
// Provides optimized BLAS (vecLib), vDSP (signal processing), and BNNS (neural networks)
//

#ifndef SD_ACCELERATEUTILS_H
#define SD_ACCELERATEUTILS_H

#include <array/NDArray.h>
#include <graph/Context.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <system/Requirements.h>

#ifdef HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

/**
 * Platform helper declarations for Apple Accelerate framework
 */

// BLAS operations (via vecLib)
DECLARE_PLATFORM(matmul, ENGINE_CPU);
DECLARE_PLATFORM(gemv, ENGINE_CPU);
DECLARE_PLATFORM(dot, ENGINE_CPU);
DECLARE_PLATFORM(axpy, ENGINE_CPU);

// Element-wise arithmetic operations (via vDSP)
DECLARE_PLATFORM(add, ENGINE_CPU);
DECLARE_PLATFORM(subtract, ENGINE_CPU);
DECLARE_PLATFORM(multiply, ENGINE_CPU);
DECLARE_PLATFORM(divide, ENGINE_CPU);

// Transcendental functions (via vDSP/vForce)
DECLARE_PLATFORM(exp, ENGINE_CPU);
DECLARE_PLATFORM(log, ENGINE_CPU);
DECLARE_PLATFORM(sqrt, ENGINE_CPU);
DECLARE_PLATFORM(abs, ENGINE_CPU);
DECLARE_PLATFORM(neg, ENGINE_CPU);
DECLARE_PLATFORM(pow, ENGINE_CPU);

// Convolution operations (via BNNS)
DECLARE_PLATFORM(conv2d, ENGINE_CPU);
DECLARE_PLATFORM(conv2d_bp, ENGINE_CPU);

// Pooling operations (via BNNS)
DECLARE_PLATFORM(avgpool2d, ENGINE_CPU);
DECLARE_PLATFORM(maxpool2d, ENGINE_CPU);

// Normalization (via BNNS)
DECLARE_PLATFORM(batchnorm, ENGINE_CPU);

// Activation functions (via BNNS/vDSP)
DECLARE_PLATFORM(relu, ENGINE_CPU);
DECLARE_PLATFORM(relu6, ENGINE_CPU);
DECLARE_PLATFORM(lrelu, ENGINE_CPU);
DECLARE_PLATFORM(elu, ENGINE_CPU);
DECLARE_PLATFORM(selu, ENGINE_CPU);
DECLARE_PLATFORM(softmax, ENGINE_CPU);
DECLARE_PLATFORM(tanh, ENGINE_CPU);
DECLARE_PLATFORM(sigmoid, ENGINE_CPU);
DECLARE_PLATFORM(swish, ENGINE_CPU);
DECLARE_PLATFORM(softplus, ENGINE_CPU);
DECLARE_PLATFORM(hardsigmoid, ENGINE_CPU);
DECLARE_PLATFORM(softsign, ENGINE_CPU);

// Trigonometric functions (via vForce)
DECLARE_PLATFORM(sin, ENGINE_CPU);
DECLARE_PLATFORM(cos, ENGINE_CPU);
DECLARE_PLATFORM(tan, ENGINE_CPU);
DECLARE_PLATFORM(asin, ENGINE_CPU);
DECLARE_PLATFORM(acos, ENGINE_CPU);
DECLARE_PLATFORM(atan, ENGINE_CPU);
DECLARE_PLATFORM(sinh, ENGINE_CPU);
DECLARE_PLATFORM(cosh, ENGINE_CPU);
DECLARE_PLATFORM(asinh, ENGINE_CPU);
DECLARE_PLATFORM(acosh, ENGINE_CPU);
DECLARE_PLATFORM(atanh, ENGINE_CPU);

// Reduction operations (via vDSP)
DECLARE_PLATFORM(reduce_sum, ENGINE_CPU);
DECLARE_PLATFORM(reduce_mean, ENGINE_CPU);
DECLARE_PLATFORM(reduce_max, ENGINE_CPU);
DECLARE_PLATFORM(reduce_min, ENGINE_CPU);
DECLARE_PLATFORM(reduce_prod, ENGINE_CPU);
DECLARE_PLATFORM(reduce_norm1, ENGINE_CPU);
DECLARE_PLATFORM(reduce_norm2, ENGINE_CPU);

// Rounding functions (via vForce)
DECLARE_PLATFORM(ceil, ENGINE_CPU);
DECLARE_PLATFORM(floor, ENGINE_CPU);
DECLARE_PLATFORM(round, ENGINE_CPU);
DECLARE_PLATFORM(rint, ENGINE_CPU);
DECLARE_PLATFORM(truncate, ENGINE_CPU);

// Additional math functions (via vForce/vDSP)
DECLARE_PLATFORM(reciprocal, ENGINE_CPU);
DECLARE_PLATFORM(rsqrt, ENGINE_CPU);
DECLARE_PLATFORM(square, ENGINE_CPU);
DECLARE_PLATFORM(log10, ENGINE_CPU);
DECLARE_PLATFORM(log1p, ENGINE_CPU);
DECLARE_PLATFORM(expm1, ENGINE_CPU);

// FFT operations (via vDSP)
DECLARE_PLATFORM(fft, ENGINE_CPU);
DECLARE_PLATFORM(ifft, ENGINE_CPU);
DECLARE_PLATFORM(rfft, ENGINE_CPU);
DECLARE_PLATFORM(irfft, ENGINE_CPU);

// Comparison and element-wise min/max (via vDSP)
DECLARE_PLATFORM(maximum, ENGINE_CPU);
DECLARE_PLATFORM(minimum, ENGINE_CPU);
DECLARE_PLATFORM(clip_by_value, ENGINE_CPU);
DECLARE_PLATFORM(sign, ENGINE_CPU);
DECLARE_PLATFORM(argmax, ENGINE_CPU);
DECLARE_PLATFORM(argmin, ENGINE_CPU);
DECLARE_PLATFORM(reduce_amax, ENGINE_CPU);

// 1D Convolution (via vDSP)
DECLARE_PLATFORM(conv1d, ENGINE_CPU);
DECLARE_PLATFORM(cross_correlate, ENGINE_CPU);
DECLARE_PLATFORM(depthwise_conv2d, ENGINE_CPU);

// Additional BLAS operations (via vecLib)
DECLARE_PLATFORM(scal, ENGINE_CPU);
DECLARE_PLATFORM(ger, ENGINE_CPU);
DECLARE_PLATFORM(copy, ENGINE_CPU);
DECLARE_PLATFORM(swap, ENGINE_CPU);
DECLARE_PLATFORM(rotg, ENGINE_CPU);
DECLARE_PLATFORM(rot, ENGINE_CPU);
DECLARE_PLATFORM(symm, ENGINE_CPU);
DECLARE_PLATFORM(syrk, ENGINE_CPU);
DECLARE_PLATFORM(trmm, ENGINE_CPU);
DECLARE_PLATFORM(trsm, ENGINE_CPU);

// Linear algebra operations (via LAPACK)
DECLARE_PLATFORM(lu, ENGINE_CPU);
DECLARE_PLATFORM(cholesky, ENGINE_CPU);
DECLARE_PLATFORM(qr, ENGINE_CPU);
DECLARE_PLATFORM(svd, ENGINE_CPU);
DECLARE_PLATFORM(solve, ENGINE_CPU);
DECLARE_PLATFORM(matrix_inverse, ENGINE_CPU);
DECLARE_PLATFORM(matrix_determinant, ENGINE_CPU);
DECLARE_PLATFORM(eigvalsh, ENGINE_CPU);

// Utility operations (via vDSP)
DECLARE_PLATFORM(reverse, ENGINE_CPU);
DECLARE_PLATFORM(sort, ENGINE_CPU);
DECLARE_PLATFORM(linspace, ENGINE_CPU);
DECLARE_PLATFORM(fill, ENGINE_CPU);
DECLARE_PLATFORM(polyval, ENGINE_CPU);
DECLARE_PLATFORM(interp, ENGINE_CPU);

// Distance operations
DECLARE_PLATFORM(hamming_distance, ENGINE_CPU);
DECLARE_PLATFORM(cosine_distance, ENGINE_CPU);
DECLARE_PLATFORM(euclidean_distance, ENGINE_CPU);
DECLARE_PLATFORM(manhattan_distance, ENGINE_CPU);

// Selection and indexing operations
DECLARE_PLATFORM(where, ENGINE_CPU);
DECLARE_PLATFORM(gather, ENGINE_CPU);
DECLARE_PLATFORM(scatter_update, ENGINE_CPU);
DECLARE_PLATFORM(scatter_add, ENGINE_CPU);
DECLARE_PLATFORM(pad, ENGINE_CPU);
DECLARE_PLATFORM(tile, ENGINE_CPU);
DECLARE_PLATFORM(concat, ENGINE_CPU);

// Gradient/backward operations
DECLARE_PLATFORM(relu_bp, ENGINE_CPU);
DECLARE_PLATFORM(sigmoid_bp, ENGINE_CPU);
DECLARE_PLATFORM(tanh_bp, ENGINE_CPU);
DECLARE_PLATFORM(elu_bp, ENGINE_CPU);
DECLARE_PLATFORM(softplus_bp, ENGINE_CPU);
DECLARE_PLATFORM(lrelu_bp, ENGINE_CPU);
DECLARE_PLATFORM(selu_bp, ENGINE_CPU);
DECLARE_PLATFORM(softmax_bp, ENGINE_CPU);
DECLARE_PLATFORM(swish_bp, ENGINE_CPU);
DECLARE_PLATFORM(hardsigmoid_bp, ENGINE_CPU);
DECLARE_PLATFORM(softsign_bp, ENGINE_CPU);

// Statistical reductions (via vDSP)
DECLARE_PLATFORM(reduce_variance, ENGINE_CPU);
DECLARE_PLATFORM(reduce_stdev, ENGINE_CPU);
DECLARE_PLATFORM(reduce_logsumexp, ENGINE_CPU);
DECLARE_PLATFORM(moments, ENGINE_CPU);

// Normalization operations
DECLARE_PLATFORM(layer_norm, ENGINE_CPU);
DECLARE_PLATFORM(layer_norm_bp, ENGINE_CPU);

// Cumulative operations (via vDSP)
DECLARE_PLATFORM(cumsum, ENGINE_CPU);
DECLARE_PLATFORM(cumprod, ENGINE_CPU);

// Batch operations (via vecLib)
DECLARE_PLATFORM(batched_gemm, ENGINE_CPU);

}  // namespace platforms

namespace accelerateUtils {

/**
 * Check if the given data type is supported by Accelerate framework
 * @param dtype The data type to check
 * @return true if the data type is supported
 */
bool isAccelerateSupported(sd::DataType dtype);

/**
 * Check if the array layout is compatible with Accelerate (contiguous, proper ordering)
 * @param arr The NDArray to check
 * @return true if the array is Accelerate-friendly
 */
bool isAccelerateFriendly(const NDArray& arr);

/**
 * Check if the array is contiguous in memory
 * @param arr The NDArray to check
 * @return true if the array is contiguous
 */
bool isContiguous(const NDArray& arr);

#ifdef HAVE_ACCELERATE
/**
 * Get BNNS data type from ND4J data type
 * @param dtype The ND4J data type
 * @return The corresponding BNNSDataType
 */
BNNSDataType getBNNSDataType(sd::DataType dtype);
#endif

/**
 * Add Accelerate requirements check to the requirements block
 * @param reqs The requirements block
 * @param block The context block
 * @param input The input NDArray (can be nullptr)
 * @param output The output NDArray (can be nullptr)
 */
void checkAccelerateRequirements(Requirements& reqs, sd::graph::Context& block,
                                  const NDArray* input = nullptr, const NDArray* output = nullptr);

/**
 * Check if Accelerate BLAS can be used for the given arrays
 * @param a First input array
 * @param b Second input array (optional)
 * @param c Output array (optional)
 * @return true if Accelerate BLAS can be used
 */
bool canUseAccelerateBlas(const NDArray& a, const NDArray* b = nullptr, const NDArray* c = nullptr);

}  // namespace accelerateUtils
}  // namespace ops
}  // namespace sd

#endif  // SD_ACCELERATEUTILS_H
