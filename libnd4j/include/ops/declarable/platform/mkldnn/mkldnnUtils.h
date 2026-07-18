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
// @author saudet
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#ifndef DEV_TESTS_MKLDNNUTILS_H
#define DEV_TESTS_MKLDNNUTILS_H

#include <array/NDArray.h>
#include <graph/Context.h>
#include <helpers/MKLDNNStream.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include <dnnl.hpp>

using namespace samediff;

#include <system/BackendNamespace.h>

namespace sd {
namespace ops {
namespace platforms {
// No namespace wrap here: DECLARE_PLATFORM/PLATFORM_IMPL open the backend
// inline namespace themselves (platform_boilerplate.h). Wrapping the list a
// second time nests SD_NS inside SD_NS and makes every unqualified SD_NS
// reference ambiguous.
/**
 * Here we actually declare our platform helpers
 */
DECLARE_PLATFORM(conv2d, ENGINE_ONEDNN);

DECLARE_PLATFORM(conv2d_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(avgpool2d, ENGINE_ONEDNN);

DECLARE_PLATFORM(avgpool2d_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(maxpool2d, ENGINE_ONEDNN);

DECLARE_PLATFORM(maxpool2d_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(conv3dnew, ENGINE_ONEDNN);

DECLARE_PLATFORM(conv3dnew_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(maxpool3dnew, ENGINE_ONEDNN);

DECLARE_PLATFORM(maxpool3dnew_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(avgpool3dnew, ENGINE_ONEDNN);

DECLARE_PLATFORM(avgpool3dnew_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(lrn, ENGINE_ONEDNN);

DECLARE_PLATFORM(batchnorm, ENGINE_ONEDNN);

DECLARE_PLATFORM(batchnorm_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(lstmLayer, ENGINE_ONEDNN);

DECLARE_PLATFORM(deconv2d, ENGINE_ONEDNN);

DECLARE_PLATFORM(deconv2d_tf, ENGINE_ONEDNN);

DECLARE_PLATFORM(deconv3d, ENGINE_ONEDNN);

DECLARE_PLATFORM(deconv2d_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(deconv3d_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(depthwise_conv2d, ENGINE_ONEDNN);

DECLARE_PLATFORM(depthwise_conv2d_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(matmul, ENGINE_ONEDNN);

DECLARE_PLATFORM(batched_gemm, ENGINE_ONEDNN);

DECLARE_PLATFORM(softmax, ENGINE_ONEDNN);

DECLARE_PLATFORM(softmax_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(tanh, ENGINE_ONEDNN);

DECLARE_PLATFORM(tanh_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(xw_plus_b, ENGINE_ONEDNN);

DECLARE_PLATFORM(xw_plus_b_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(concat, ENGINE_ONEDNN);

// Activation functions
DECLARE_PLATFORM(relu, ENGINE_ONEDNN);
DECLARE_PLATFORM(relu_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(sigmoid, ENGINE_ONEDNN);
DECLARE_PLATFORM(sigmoid_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(elu, ENGINE_ONEDNN);
DECLARE_PLATFORM(elu_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(lrelu, ENGINE_ONEDNN);
DECLARE_PLATFORM(lrelu_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(relu6, ENGINE_ONEDNN);
DECLARE_PLATFORM(relu6_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(softplus, ENGINE_ONEDNN);
DECLARE_PLATFORM(softplus_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(hardsigmoid, ENGINE_ONEDNN);

DECLARE_PLATFORM(mish, ENGINE_ONEDNN);
DECLARE_PLATFORM(mish_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(Abs, ENGINE_ONEDNN);

DECLARE_PLATFORM(log_softmax, ENGINE_ONEDNN);
DECLARE_PLATFORM(log_softmax_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(swish, ENGINE_ONEDNN);
DECLARE_PLATFORM(swish_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(gelu, ENGINE_ONEDNN);
DECLARE_PLATFORM(gelu_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(hardtanh, ENGINE_ONEDNN);
DECLARE_PLATFORM(hardtanh_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(selu, ENGINE_ONEDNN);
DECLARE_PLATFORM(selu_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(hardswish, ENGINE_ONEDNN);
DECLARE_PLATFORM(hardswish_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(logsigmoid, ENGINE_ONEDNN);

// Math operations
DECLARE_PLATFORM(square, ENGINE_ONEDNN);

DECLARE_PLATFORM(Exp, ENGINE_ONEDNN);

DECLARE_PLATFORM(Log, ENGINE_ONEDNN);

DECLARE_PLATFORM(Sqrt, ENGINE_ONEDNN);

DECLARE_PLATFORM(Pow, ENGINE_ONEDNN);

DECLARE_PLATFORM(Round, ENGINE_ONEDNN);

DECLARE_PLATFORM(Neg, ENGINE_ONEDNN);

// Error functions (MKL VML accelerated)
DECLARE_PLATFORM(Erf, ENGINE_ONEDNN);
DECLARE_PLATFORM(Erfc, ENGINE_ONEDNN);

DECLARE_PLATFORM(clipbyvalue, ENGINE_ONEDNN);
DECLARE_PLATFORM(clipbyvalue_bp, ENGINE_ONEDNN);

DECLARE_PLATFORM(thresholdedrelu, ENGINE_ONEDNN);
DECLARE_PLATFORM(thresholdedrelu_bp, ENGINE_ONEDNN);

// Binary operations
DECLARE_PLATFORM(add, ENGINE_ONEDNN);
DECLARE_PLATFORM(subtract, ENGINE_ONEDNN);
DECLARE_PLATFORM(multiply, ENGINE_ONEDNN);
DECLARE_PLATFORM(divide, ENGINE_ONEDNN);
DECLARE_PLATFORM(maximum, ENGINE_ONEDNN);
DECLARE_PLATFORM(minimum, ENGINE_ONEDNN);

// Reduction operations
DECLARE_PLATFORM(reduce_sum, ENGINE_ONEDNN);
DECLARE_PLATFORM(reduce_mean, ENGINE_ONEDNN);
DECLARE_PLATFORM(reduce_max, ENGINE_ONEDNN);
DECLARE_PLATFORM(reduce_min, ENGINE_ONEDNN);
DECLARE_PLATFORM(reduce_prod, ENGINE_ONEDNN);

// Normalization
DECLARE_PLATFORM(layer_norm, ENGINE_ONEDNN);

// PReLU (Parametric ReLU)
DECLARE_PLATFORM(prelu, ENGINE_ONEDNN);
DECLARE_PLATFORM(prelu_bp, ENGINE_ONEDNN);

// Multi-tensor sum
DECLARE_PLATFORM(mergeadd, ENGINE_ONEDNN);

// Resampling/Resize
DECLARE_PLATFORM(resize_bilinear, ENGINE_ONEDNN);
DECLARE_PLATFORM(resize_nearest_neighbor, ENGINE_ONEDNN);

// Dense/Inner product
DECLARE_PLATFORM(dense, ENGINE_ONEDNN);

// Extended eltwise operations
DECLARE_PLATFORM(expm1, ENGINE_ONEDNN);
DECLARE_PLATFORM(log1p, ENGINE_ONEDNN);
DECLARE_PLATFORM(gelu_tanh, ENGINE_ONEDNN);

// Global pooling
DECLARE_PLATFORM(global_max_pooling_2d, ENGINE_ONEDNN);
DECLARE_PLATFORM(global_avg_pooling_2d, ENGINE_ONEDNN);

// Channel shuffle
DECLARE_PLATFORM(shuffle_channel, ENGINE_ONEDNN);
DECLARE_PLATFORM(shuffle_channel_bp, ENGINE_ONEDNN);

// Backward passes for math operations
DECLARE_PLATFORM(square_bp, ENGINE_ONEDNN);
DECLARE_PLATFORM(Exp_bp, ENGINE_ONEDNN);
DECLARE_PLATFORM(Log_bp, ENGINE_ONEDNN);
DECLARE_PLATFORM(Sqrt_bp, ENGINE_ONEDNN);
DECLARE_PLATFORM(Abs_bp, ENGINE_ONEDNN);
DECLARE_PLATFORM(hardsigmoid_bp, ENGINE_ONEDNN);

// Hyperbolic functions
DECLARE_PLATFORM(cosh, ENGINE_ONEDNN);
DECLARE_PLATFORM(sinh, ENGINE_ONEDNN);

// Transpose/Permute
DECLARE_PLATFORM(transpose, ENGINE_ONEDNN);
DECLARE_PLATFORM(permute, ENGINE_ONEDNN);

// 1D Convolution
DECLARE_PLATFORM(conv1d, ENGINE_ONEDNN);
DECLARE_PLATFORM(conv1d_bp, ENGINE_ONEDNN);
DECLARE_PLATFORM(deconv1d, ENGINE_ONEDNN);

// 1D Pooling
DECLARE_PLATFORM(maxpool1d, ENGINE_ONEDNN);
DECLARE_PLATFORM(maxpool1d_bp, ENGINE_ONEDNN);
DECLARE_PLATFORM(avgpool1d, ENGINE_ONEDNN);
DECLARE_PLATFORM(avgpool1d_bp, ENGINE_ONEDNN);

// Comparison operations
DECLARE_PLATFORM(greater, ENGINE_ONEDNN);
DECLARE_PLATFORM(greater_equal, ENGINE_ONEDNN);
DECLARE_PLATFORM(less, ENGINE_ONEDNN);
DECLARE_PLATFORM(less_equal, ENGINE_ONEDNN);
DECLARE_PLATFORM(equals, ENGINE_ONEDNN);
DECLARE_PLATFORM(not_equals, ENGINE_ONEDNN);

// Scale and add operations
DECLARE_PLATFORM(add_scalar, ENGINE_ONEDNN);
DECLARE_PLATFORM(multiply_scalar, ENGINE_ONEDNN);
DECLARE_PLATFORM(axpy, ENGINE_ONEDNN);

// GRU/RNN operations
DECLARE_PLATFORM(gruCell, ENGINE_ONEDNN);

// Miscellaneous eltwise operations
DECLARE_PLATFORM(squaredsubtract, ENGINE_ONEDNN);
DECLARE_PLATFORM(rsqrt, ENGINE_ONEDNN);
DECLARE_PLATFORM(Reciprocal, ENGINE_ONEDNN);
DECLARE_PLATFORM(cube, ENGINE_ONEDNN);
DECLARE_PLATFORM(identity, ENGINE_ONEDNN);

// Statistical moments
DECLARE_PLATFORM(reduce_variance, ENGINE_ONEDNN);
DECLARE_PLATFORM(reduce_stdev, ENGINE_ONEDNN);

// Split operations
DECLARE_PLATFORM(split, ENGINE_ONEDNN);
DECLARE_PLATFORM(split_v, ENGINE_ONEDNN);

// Stack/unstack operations
DECLARE_PLATFORM(stack, ENGINE_ONEDNN);
DECLARE_PLATFORM(unstack, ENGINE_ONEDNN);

// Reverse operations
DECLARE_PLATFORM(reverse, ENGINE_ONEDNN);
DECLARE_PLATFORM(reverse_sequence, ENGINE_ONEDNN);

// Sign operation
DECLARE_PLATFORM(Sign, ENGINE_ONEDNN);

// Gather operations
DECLARE_PLATFORM(gather, ENGINE_ONEDNN);
DECLARE_PLATFORM(gather_nd, ENGINE_ONEDNN);

// Tile operation
DECLARE_PLATFORM(tile, ENGINE_ONEDNN);

// Broadcast operations
DECLARE_PLATFORM(broadcast_to, ENGINE_ONEDNN);

// Pad operations
DECLARE_PLATFORM(pad, ENGINE_ONEDNN);
DECLARE_PLATFORM(mirror_pad, ENGINE_ONEDNN);

// Scatter operations
DECLARE_PLATFORM(scatter_update, ENGINE_ONEDNN);
DECLARE_PLATFORM(scatter_add, ENGINE_ONEDNN);
DECLARE_PLATFORM(scatter_nd, ENGINE_ONEDNN);

// Embedding operations
DECLARE_PLATFORM(embedding_lookup, ENGINE_ONEDNN);
DECLARE_PLATFORM(segment_sum, ENGINE_ONEDNN);
DECLARE_PLATFORM(segment_mean, ENGINE_ONEDNN);

// One-hot encoding
DECLARE_PLATFORM(onehot, ENGINE_ONEDNN);

// ArgMax/ArgMin operations
DECLARE_PLATFORM(argmax, ENGINE_ONEDNN);
DECLARE_PLATFORM(argmin, ENGINE_ONEDNN);

// Select/Where operations
DECLARE_PLATFORM(select, ENGINE_ONEDNN);
DECLARE_PLATFORM(choose, ENGINE_ONEDNN);

// Reshape operations
DECLARE_PLATFORM(reshape, ENGINE_ONEDNN);
DECLARE_PLATFORM(squeeze, ENGINE_ONEDNN);
DECLARE_PLATFORM(expand_dims, ENGINE_ONEDNN);
DECLARE_PLATFORM(flatten, ENGINE_ONEDNN);
DECLARE_PLATFORM(flatten_2d, ENGINE_ONEDNN);

// Slice operations
DECLARE_PLATFORM(slice, ENGINE_ONEDNN);
DECLARE_PLATFORM(strided_slice, ENGINE_ONEDNN);

// Space transformation operations
DECLARE_PLATFORM(depth_to_space, ENGINE_ONEDNN);
DECLARE_PLATFORM(space_to_depth, ENGINE_ONEDNN);
DECLARE_PLATFORM(batch_to_space, ENGINE_ONEDNN);
DECLARE_PLATFORM(space_to_batch, ENGINE_ONEDNN);

// Fill/Range operations
DECLARE_PLATFORM(fill, ENGINE_ONEDNN);
DECLARE_PLATFORM(zeros_like, ENGINE_ONEDNN);
DECLARE_PLATFORM(ones_like, ENGINE_ONEDNN);
DECLARE_PLATFORM(range, ENGINE_ONEDNN);
DECLARE_PLATFORM(lin_space, ENGINE_ONEDNN);
DECLARE_PLATFORM(eye, ENGINE_ONEDNN);

// Norm operations
DECLARE_PLATFORM(reduce_norm1, ENGINE_ONEDNN);
DECLARE_PLATFORM(reduce_norm2, ENGINE_ONEDNN);
DECLARE_PLATFORM(reduce_norm_max, ENGINE_ONEDNN);

// Boolean operations
DECLARE_PLATFORM(boolean_and, ENGINE_ONEDNN);
DECLARE_PLATFORM(boolean_or, ENGINE_ONEDNN);
DECLARE_PLATFORM(boolean_xor, ENGINE_ONEDNN);
DECLARE_PLATFORM(boolean_not, ENGINE_ONEDNN);
DECLARE_PLATFORM(isfinite, ENGINE_ONEDNN);
DECLARE_PLATFORM(isinf, ENGINE_ONEDNN);
DECLARE_PLATFORM(isnan, ENGINE_ONEDNN);

// TopK and Sort operations
DECLARE_PLATFORM(top_k, ENGINE_ONEDNN);
DECLARE_PLATFORM(in_top_k, ENGINE_ONEDNN);
DECLARE_PLATFORM(sort, ENGINE_ONEDNN);
DECLARE_PLATFORM(argsort, ENGINE_ONEDNN);

// Cumulative operations
DECLARE_PLATFORM(cumsum, ENGINE_ONEDNN);
DECLARE_PLATFORM(cumprod, ENGINE_ONEDNN);
DECLARE_PLATFORM(cummax, ENGINE_ONEDNN);
DECLARE_PLATFORM(cummin, ENGINE_ONEDNN);

// Diagonal operations
DECLARE_PLATFORM(diag, ENGINE_ONEDNN);
DECLARE_PLATFORM(diag_part, ENGINE_ONEDNN);
DECLARE_PLATFORM(trace, ENGINE_ONEDNN);
DECLARE_PLATFORM(triu, ENGINE_ONEDNN);
DECLARE_PLATFORM(tril, ENGINE_ONEDNN);

// Loss operations
DECLARE_PLATFORM(softmax_cross_entropy_loss_with_logits, ENGINE_ONEDNN);
DECLARE_PLATFORM(sigmoid_cross_entropy_loss_with_logits, ENGINE_ONEDNN);
DECLARE_PLATFORM(mean_sqerr_loss, ENGINE_ONEDNN);
DECLARE_PLATFORM(huber_loss, ENGINE_ONEDNN);
DECLARE_PLATFORM(log_loss, ENGINE_ONEDNN);

// Dropout and random operations
DECLARE_PLATFORM(dropout, ENGINE_ONEDNN);
DECLARE_PLATFORM(alpha_dropout, ENGINE_ONEDNN);
DECLARE_PLATFORM(random_normal, ENGINE_ONEDNN);
DECLARE_PLATFORM(random_uniform, ENGINE_ONEDNN);
DECLARE_PLATFORM(random_bernoulli, ENGINE_ONEDNN);

// Grid operations
DECLARE_PLATFORM(meshgrid, ENGINE_ONEDNN);
DECLARE_PLATFORM(repeat, ENGINE_ONEDNN);
DECLARE_PLATFORM(roll, ENGINE_ONEDNN);

// Attention operations - Scaled Dot Product Attention (SDPA)
DECLARE_PLATFORM(dot_product_attention_v2, ENGINE_ONEDNN);
DECLARE_PLATFORM(dot_product_attention_v2_bp, ENGINE_ONEDNN);

// Flash Attention - memory efficient attention with 3D/4D support
DECLARE_PLATFORM(flash_attention, ENGINE_ONEDNN);
DECLARE_PLATFORM(flash_attention_bp, ENGINE_ONEDNN);

}  // namespace platforms
}  // namespace ops

SD_BACKEND_ROOT_INLINE_NAMESPACE_BEGIN
namespace onednnUtils {

void poolingONEDNN(NDArray* input, NDArray* output, const sd::LongType kD, const sd::LongType kH, const sd::LongType kW, const sd::LongType sD,
                   const sd::LongType sH, const sd::LongType sW, const sd::LongType pD, const sd::LongType pH, const sd::LongType pW, const int isNCHW,
                   const dnnl::algorithm mode);

void poolingBpONEDNN(NDArray* input, NDArray* gradO, NDArray* gradI, const sd::LongType kD, const sd::LongType kH,
                     const sd::LongType kW, const sd::LongType sD, const sd::LongType sH, const sd::LongType sW, const sd::LongType pD, const sd::LongType pH, const sd::LongType pW,
                     const int isNCHW, const dnnl::algorithm mode);

void getONEDNNMemoryDescLrn(NDArray* src, NDArray* diff_src, NDArray* dst,
                            dnnl::memory::desc* lrn_src_md, dnnl::memory::desc* lrn_diff_src_md,
                            dnnl::memory::desc* lrn_dst_md, dnnl::memory::desc* user_src_md,
                            dnnl::memory::desc* user_diff_src_md, dnnl::memory::desc* user_dst_md, int axis);

dnnl::engine& getEngine(void* ptr);

/**
 * This function creates memory dimentions
 * @param const pointer to array
 * @param const array rank
 * @param reference to memory dimentions
 */
void getDims(NDArray* array, const int rank, dnnl::memory::dims& mklDims);
/**
 * This function evaluate memory format tag based on array shapeInfo
 * @param const array
 * @return memory format
 */
dnnl::memory::format_tag getFormat(NDArray& arr);

void setBlockStrides(NDArray& array, dnnl::memory::desc& mklMd, const std::vector<int>& permut = {});
//////////////////////////////////////////////////////////////////////
/**
 * This function load and reorder user memory to mkl
 * @param const pointer to dataset
 * @param reference to mkl engine
 * @param reference to mkl stream
 * @param reference to args container for dnnl
 * @param reference to user memory description
 * @param primitive memory descriptor
 * @param dnnl arg activation enumerator
 */
dnnl::memory loadDataToMklStream(NDArray& array, const dnnl::engine& engine, const dnnl::stream& stream,
                                 const dnnl::memory::desc& user_md, const dnnl::memory::desc& primitive_md,
                                 dnnl::memory& arg);

/**
 * @brief This function checks adittional ONEDNN pooling requirements
 *
 * @param reqs Requirements block to store the check result
 * @param block Context block to extract positional integer arguments.
 * @param in in NDArray
 * @param out out NDArray
 */
void checkPoolingONEDNN(Requirements& reqs, sd::graph::Context& block, sd::NDArray* in, sd::NDArray* out);



}  // namespace onednnUtils
SD_BACKEND_ROOT_INLINE_NAMESPACE_END
}  // namespace sd

#endif  // DEV_TESTS_MKLDNNUTILS_H
