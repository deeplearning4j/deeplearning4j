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

namespace sd {
namespace ops {
namespace platforms {
/**
 * Here we actually declare our platform helpers
 */
DECLARE_PLATFORM(conv2d, ENGINE_CPU);

DECLARE_PLATFORM(conv2d_bp, ENGINE_CPU);

DECLARE_PLATFORM(avgpool2d, ENGINE_CPU);

DECLARE_PLATFORM(avgpool2d_bp, ENGINE_CPU);

DECLARE_PLATFORM(maxpool2d, ENGINE_CPU);

DECLARE_PLATFORM(maxpool2d_bp, ENGINE_CPU);

DECLARE_PLATFORM(conv3dnew, ENGINE_CPU);

DECLARE_PLATFORM(conv3dnew_bp, ENGINE_CPU);

DECLARE_PLATFORM(maxpool3dnew, ENGINE_CPU);

DECLARE_PLATFORM(maxpool3dnew_bp, ENGINE_CPU);

DECLARE_PLATFORM(avgpool3dnew, ENGINE_CPU);

DECLARE_PLATFORM(avgpool3dnew_bp, ENGINE_CPU);

DECLARE_PLATFORM(lrn, ENGINE_CPU);

DECLARE_PLATFORM(batchnorm, ENGINE_CPU);

DECLARE_PLATFORM(batchnorm_bp, ENGINE_CPU);

DECLARE_PLATFORM(lstmLayer, ENGINE_CPU);

DECLARE_PLATFORM(deconv2d, ENGINE_CPU);

DECLARE_PLATFORM(deconv2d_tf, ENGINE_CPU);

DECLARE_PLATFORM(deconv3d, ENGINE_CPU);

DECLARE_PLATFORM(deconv2d_bp, ENGINE_CPU);

DECLARE_PLATFORM(deconv3d_bp, ENGINE_CPU);

DECLARE_PLATFORM(depthwise_conv2d, ENGINE_CPU);

DECLARE_PLATFORM(depthwise_conv2d_bp, ENGINE_CPU);

DECLARE_PLATFORM(matmul, ENGINE_CPU);

DECLARE_PLATFORM(batched_gemm, ENGINE_CPU);

DECLARE_PLATFORM(softmax, ENGINE_CPU);

DECLARE_PLATFORM(softmax_bp, ENGINE_CPU);

DECLARE_PLATFORM(tanh, ENGINE_CPU);

DECLARE_PLATFORM(tanh_bp, ENGINE_CPU);

DECLARE_PLATFORM(xw_plus_b, ENGINE_CPU);

DECLARE_PLATFORM(xw_plus_b_bp, ENGINE_CPU);

DECLARE_PLATFORM(concat, ENGINE_CPU);

// Activation functions
DECLARE_PLATFORM(relu, ENGINE_CPU);
DECLARE_PLATFORM(relu_bp, ENGINE_CPU);

DECLARE_PLATFORM(sigmoid, ENGINE_CPU);
DECLARE_PLATFORM(sigmoid_bp, ENGINE_CPU);

DECLARE_PLATFORM(elu, ENGINE_CPU);
DECLARE_PLATFORM(elu_bp, ENGINE_CPU);

DECLARE_PLATFORM(lrelu, ENGINE_CPU);
DECLARE_PLATFORM(lrelu_bp, ENGINE_CPU);

DECLARE_PLATFORM(relu6, ENGINE_CPU);
DECLARE_PLATFORM(relu6_bp, ENGINE_CPU);

DECLARE_PLATFORM(softplus, ENGINE_CPU);
DECLARE_PLATFORM(softplus_bp, ENGINE_CPU);

DECLARE_PLATFORM(hardsigmoid, ENGINE_CPU);

DECLARE_PLATFORM(mish, ENGINE_CPU);
DECLARE_PLATFORM(mish_bp, ENGINE_CPU);

DECLARE_PLATFORM(Abs, ENGINE_CPU);

DECLARE_PLATFORM(log_softmax, ENGINE_CPU);
DECLARE_PLATFORM(log_softmax_bp, ENGINE_CPU);

DECLARE_PLATFORM(swish, ENGINE_CPU);
DECLARE_PLATFORM(swish_bp, ENGINE_CPU);

DECLARE_PLATFORM(gelu, ENGINE_CPU);
DECLARE_PLATFORM(gelu_bp, ENGINE_CPU);

DECLARE_PLATFORM(hardtanh, ENGINE_CPU);
DECLARE_PLATFORM(hardtanh_bp, ENGINE_CPU);

DECLARE_PLATFORM(selu, ENGINE_CPU);
DECLARE_PLATFORM(selu_bp, ENGINE_CPU);

DECLARE_PLATFORM(hardswish, ENGINE_CPU);
DECLARE_PLATFORM(hardswish_bp, ENGINE_CPU);

DECLARE_PLATFORM(logsigmoid, ENGINE_CPU);

// Math operations
DECLARE_PLATFORM(square, ENGINE_CPU);

DECLARE_PLATFORM(Exp, ENGINE_CPU);

DECLARE_PLATFORM(Log, ENGINE_CPU);

DECLARE_PLATFORM(Sqrt, ENGINE_CPU);

DECLARE_PLATFORM(Pow, ENGINE_CPU);

DECLARE_PLATFORM(Round, ENGINE_CPU);

DECLARE_PLATFORM(Neg, ENGINE_CPU);

// Error functions (MKL VML accelerated)
DECLARE_PLATFORM(Erf, ENGINE_CPU);
DECLARE_PLATFORM(Erfc, ENGINE_CPU);

DECLARE_PLATFORM(clipbyvalue, ENGINE_CPU);
DECLARE_PLATFORM(clipbyvalue_bp, ENGINE_CPU);

DECLARE_PLATFORM(thresholdedrelu, ENGINE_CPU);
DECLARE_PLATFORM(thresholdedrelu_bp, ENGINE_CPU);

// Binary operations
DECLARE_PLATFORM(add, ENGINE_CPU);
DECLARE_PLATFORM(subtract, ENGINE_CPU);
DECLARE_PLATFORM(multiply, ENGINE_CPU);
DECLARE_PLATFORM(divide, ENGINE_CPU);
DECLARE_PLATFORM(maximum, ENGINE_CPU);
DECLARE_PLATFORM(minimum, ENGINE_CPU);

// Reduction operations
DECLARE_PLATFORM(reduce_sum, ENGINE_CPU);
DECLARE_PLATFORM(reduce_mean, ENGINE_CPU);
DECLARE_PLATFORM(reduce_max, ENGINE_CPU);
DECLARE_PLATFORM(reduce_min, ENGINE_CPU);
DECLARE_PLATFORM(reduce_prod, ENGINE_CPU);

// Normalization
DECLARE_PLATFORM(layer_norm, ENGINE_CPU);

// PReLU (Parametric ReLU)
DECLARE_PLATFORM(prelu, ENGINE_CPU);
DECLARE_PLATFORM(prelu_bp, ENGINE_CPU);

// Multi-tensor sum
DECLARE_PLATFORM(mergeadd, ENGINE_CPU);

// Resampling/Resize
DECLARE_PLATFORM(resize_bilinear, ENGINE_CPU);
DECLARE_PLATFORM(resize_nearest_neighbor, ENGINE_CPU);

// Dense/Inner product
DECLARE_PLATFORM(dense, ENGINE_CPU);

// Extended eltwise operations
DECLARE_PLATFORM(expm1, ENGINE_CPU);
DECLARE_PLATFORM(log1p, ENGINE_CPU);
DECLARE_PLATFORM(gelu_tanh, ENGINE_CPU);

// Global pooling
DECLARE_PLATFORM(global_max_pooling_2d, ENGINE_CPU);
DECLARE_PLATFORM(global_avg_pooling_2d, ENGINE_CPU);

// Channel shuffle
DECLARE_PLATFORM(shuffle_channel, ENGINE_CPU);
DECLARE_PLATFORM(shuffle_channel_bp, ENGINE_CPU);

// Backward passes for math operations
DECLARE_PLATFORM(square_bp, ENGINE_CPU);
DECLARE_PLATFORM(Exp_bp, ENGINE_CPU);
DECLARE_PLATFORM(Log_bp, ENGINE_CPU);
DECLARE_PLATFORM(Sqrt_bp, ENGINE_CPU);
DECLARE_PLATFORM(Abs_bp, ENGINE_CPU);
DECLARE_PLATFORM(hardsigmoid_bp, ENGINE_CPU);

// Hyperbolic functions
DECLARE_PLATFORM(cosh, ENGINE_CPU);
DECLARE_PLATFORM(sinh, ENGINE_CPU);

// Transpose/Permute
DECLARE_PLATFORM(transpose, ENGINE_CPU);
DECLARE_PLATFORM(permute, ENGINE_CPU);

// 1D Convolution
DECLARE_PLATFORM(conv1d, ENGINE_CPU);
DECLARE_PLATFORM(conv1d_bp, ENGINE_CPU);
DECLARE_PLATFORM(deconv1d, ENGINE_CPU);

// 1D Pooling
DECLARE_PLATFORM(maxpool1d, ENGINE_CPU);
DECLARE_PLATFORM(maxpool1d_bp, ENGINE_CPU);
DECLARE_PLATFORM(avgpool1d, ENGINE_CPU);
DECLARE_PLATFORM(avgpool1d_bp, ENGINE_CPU);

// Comparison operations
DECLARE_PLATFORM(greater, ENGINE_CPU);
DECLARE_PLATFORM(greater_equal, ENGINE_CPU);
DECLARE_PLATFORM(less, ENGINE_CPU);
DECLARE_PLATFORM(less_equal, ENGINE_CPU);
DECLARE_PLATFORM(equals, ENGINE_CPU);
DECLARE_PLATFORM(not_equals, ENGINE_CPU);

// Scale and add operations
DECLARE_PLATFORM(add_scalar, ENGINE_CPU);
DECLARE_PLATFORM(multiply_scalar, ENGINE_CPU);
DECLARE_PLATFORM(axpy, ENGINE_CPU);

// GRU/RNN operations
DECLARE_PLATFORM(gruCell, ENGINE_CPU);

// Miscellaneous eltwise operations
DECLARE_PLATFORM(squaredsubtract, ENGINE_CPU);
DECLARE_PLATFORM(rsqrt, ENGINE_CPU);
DECLARE_PLATFORM(Reciprocal, ENGINE_CPU);
DECLARE_PLATFORM(cube, ENGINE_CPU);
DECLARE_PLATFORM(identity, ENGINE_CPU);

// Statistical moments
DECLARE_PLATFORM(reduce_variance, ENGINE_CPU);
DECLARE_PLATFORM(reduce_stdev, ENGINE_CPU);

// Split operations
DECLARE_PLATFORM(split, ENGINE_CPU);
DECLARE_PLATFORM(split_v, ENGINE_CPU);

// Stack/unstack operations
DECLARE_PLATFORM(stack, ENGINE_CPU);
DECLARE_PLATFORM(unstack, ENGINE_CPU);

// Reverse operations
DECLARE_PLATFORM(reverse, ENGINE_CPU);
DECLARE_PLATFORM(reverse_sequence, ENGINE_CPU);

// Sign operation
DECLARE_PLATFORM(Sign, ENGINE_CPU);

// Gather operations
DECLARE_PLATFORM(gather, ENGINE_CPU);
DECLARE_PLATFORM(gather_nd, ENGINE_CPU);

// Tile operation
DECLARE_PLATFORM(tile, ENGINE_CPU);

// Broadcast operations
DECLARE_PLATFORM(broadcast_to, ENGINE_CPU);

// Pad operations
DECLARE_PLATFORM(pad, ENGINE_CPU);
DECLARE_PLATFORM(mirror_pad, ENGINE_CPU);

// Scatter operations
DECLARE_PLATFORM(scatter_update, ENGINE_CPU);
DECLARE_PLATFORM(scatter_add, ENGINE_CPU);
DECLARE_PLATFORM(scatter_nd, ENGINE_CPU);

// Embedding operations
DECLARE_PLATFORM(embedding_lookup, ENGINE_CPU);
DECLARE_PLATFORM(segment_sum, ENGINE_CPU);
DECLARE_PLATFORM(segment_mean, ENGINE_CPU);

// One-hot encoding
DECLARE_PLATFORM(onehot, ENGINE_CPU);

// ArgMax/ArgMin operations
DECLARE_PLATFORM(argmax, ENGINE_CPU);
DECLARE_PLATFORM(argmin, ENGINE_CPU);

// Select/Where operations
DECLARE_PLATFORM(select, ENGINE_CPU);
DECLARE_PLATFORM(choose, ENGINE_CPU);

// Reshape operations
DECLARE_PLATFORM(reshape, ENGINE_CPU);
DECLARE_PLATFORM(squeeze, ENGINE_CPU);
DECLARE_PLATFORM(expand_dims, ENGINE_CPU);
DECLARE_PLATFORM(flatten, ENGINE_CPU);
DECLARE_PLATFORM(flatten_2d, ENGINE_CPU);

// Slice operations
DECLARE_PLATFORM(slice, ENGINE_CPU);
DECLARE_PLATFORM(strided_slice, ENGINE_CPU);

// Space transformation operations
DECLARE_PLATFORM(depth_to_space, ENGINE_CPU);
DECLARE_PLATFORM(space_to_depth, ENGINE_CPU);
DECLARE_PLATFORM(batch_to_space, ENGINE_CPU);
DECLARE_PLATFORM(space_to_batch, ENGINE_CPU);

// Fill/Range operations
DECLARE_PLATFORM(fill, ENGINE_CPU);
DECLARE_PLATFORM(zeros_like, ENGINE_CPU);
DECLARE_PLATFORM(ones_like, ENGINE_CPU);
DECLARE_PLATFORM(range, ENGINE_CPU);
DECLARE_PLATFORM(lin_space, ENGINE_CPU);
DECLARE_PLATFORM(eye, ENGINE_CPU);

// Norm operations
DECLARE_PLATFORM(reduce_norm1, ENGINE_CPU);
DECLARE_PLATFORM(reduce_norm2, ENGINE_CPU);
DECLARE_PLATFORM(reduce_norm_max, ENGINE_CPU);

// Boolean operations
DECLARE_PLATFORM(boolean_and, ENGINE_CPU);
DECLARE_PLATFORM(boolean_or, ENGINE_CPU);
DECLARE_PLATFORM(boolean_xor, ENGINE_CPU);
DECLARE_PLATFORM(boolean_not, ENGINE_CPU);
DECLARE_PLATFORM(isfinite, ENGINE_CPU);
DECLARE_PLATFORM(isinf, ENGINE_CPU);
DECLARE_PLATFORM(isnan, ENGINE_CPU);

// TopK and Sort operations
DECLARE_PLATFORM(top_k, ENGINE_CPU);
DECLARE_PLATFORM(in_top_k, ENGINE_CPU);
DECLARE_PLATFORM(sort, ENGINE_CPU);
DECLARE_PLATFORM(argsort, ENGINE_CPU);

// Cumulative operations
DECLARE_PLATFORM(cumsum, ENGINE_CPU);
DECLARE_PLATFORM(cumprod, ENGINE_CPU);
DECLARE_PLATFORM(cummax, ENGINE_CPU);
DECLARE_PLATFORM(cummin, ENGINE_CPU);

// Diagonal operations
DECLARE_PLATFORM(diag, ENGINE_CPU);
DECLARE_PLATFORM(diag_part, ENGINE_CPU);
DECLARE_PLATFORM(trace, ENGINE_CPU);
DECLARE_PLATFORM(triu, ENGINE_CPU);
DECLARE_PLATFORM(tril, ENGINE_CPU);

// Loss operations
DECLARE_PLATFORM(softmax_cross_entropy_loss_with_logits, ENGINE_CPU);
DECLARE_PLATFORM(sigmoid_cross_entropy_loss_with_logits, ENGINE_CPU);
DECLARE_PLATFORM(mean_sqerr_loss, ENGINE_CPU);
DECLARE_PLATFORM(huber_loss, ENGINE_CPU);
DECLARE_PLATFORM(log_loss, ENGINE_CPU);

// Dropout and random operations
DECLARE_PLATFORM(dropout, ENGINE_CPU);
DECLARE_PLATFORM(alpha_dropout, ENGINE_CPU);
DECLARE_PLATFORM(random_normal, ENGINE_CPU);
DECLARE_PLATFORM(random_uniform, ENGINE_CPU);
DECLARE_PLATFORM(random_bernoulli, ENGINE_CPU);

// Grid operations
DECLARE_PLATFORM(meshgrid, ENGINE_CPU);
DECLARE_PLATFORM(repeat, ENGINE_CPU);
DECLARE_PLATFORM(roll, ENGINE_CPU);

// Attention operations - Scaled Dot Product Attention (SDPA)
DECLARE_PLATFORM(dot_product_attention_v2, ENGINE_CPU);
DECLARE_PLATFORM(dot_product_attention_v2_bp, ENGINE_CPU);

// Flash Attention - memory efficient attention with 3D/4D support
DECLARE_PLATFORM(flash_attention, ENGINE_CPU);
DECLARE_PLATFORM(flash_attention_bp, ENGINE_CPU);

}  // namespace platforms
}  // namespace ops

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
}  // namespace sd

#endif  // DEV_TESTS_MKLDNNUTILS_H
