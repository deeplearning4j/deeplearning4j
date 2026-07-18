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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <helpers/MKLDNNStream.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/helpers/convolutions.h>
#include <system/platform_boilerplate.h>

#include "mkldnnUtils.h"

using namespace dnnl;

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////
// Get OneDNN data type from NDArray
static dnnl::memory::data_type getOneDnnDataType(DataType dt) {
  switch (dt) {
    case DataType::FLOAT32:
      return dnnl::memory::data_type::f32;
    case DataType::BFLOAT16:
      return dnnl::memory::data_type::bf16;
    case DataType::HALF:
      return dnnl::memory::data_type::f16;
    default:
      return dnnl::memory::data_type::f32;
  }
}

//////////////////////////////////////////////////////////////////////
// Check if data type is supported by OneDNN conv2d (runtime ISA detection)
static bool isSupportedConv2dType(DataType dt) {
  if (dt == DataType::FLOAT32) return true;
  if (dt == DataType::BFLOAT16) {
    dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
    return (isa >= dnnl_cpu_isa_avx512_core_bf16);
  }
  if (dt == DataType::HALF) {
    dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
    return (isa >= dnnl_cpu_isa_avx512_core_amx_fp16);
  }
  return false;
}

//////////////////////////////////////////////////////////////////////
static void conv2dMKLDNN(NDArray *input, NDArray *weights, NDArray *bias, NDArray *output,
                         const sd::LongType kH, const sd::LongType kW, const sd::LongType sH, const sd::LongType sW, const sd::LongType pH, const sd::LongType pW,
                         const sd::LongType dH, const sd::LongType dW, const int paddingMode, const int isNCHW, const int wFormat) {
  // mkl support weights in [oC, iC, kH, kW] format only

  sd::LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  sd::LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWoC, indWkH, indOoH);

  sd_debug("Running conv2d onednn with strides: %d %d padding: %d %d dilation: %d %d paddingMode %d weightFormat %d\n",sH,sW,pH,pW,dH,dW,paddingMode,wFormat);
  const int pWSame = (paddingMode == 2 && dW > 1) ? ((oW - 1) * sW + (kW - 1) * dW + 1 - iW) / 2
                                                  : pW;  // dH == 1 for causal mode in conv1d

  // Effective kernel sizes accounting for dilation. OneDNN padding_r must be computed
  // using the effective (dilated) kernel size, not the raw kernel size. Using raw kH/kW
  // gives negative padding_r when dilation > 1 with SAME padding, causing OneDNN to reject
  // the primitive descriptor.
  const sd::LongType eKH = (kH - 1) * dH + 1;
  const sd::LongType eKW = (kW - 1) * dW + 1;

  dnnl::memory::dims strides = {sH, sW};
  dnnl::memory::dims padding = {pH, pW};
  dnnl::memory::dims padding_r = {(oH - 1) * sH - iH + eKH - pH, (oW - 1) * sW - iW + eKW - pWSame};
  dnnl::memory::dims dilation = {dH - 1, dW - 1};

  auto xzFormatMkl = isNCHW ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::nhwc;
  dnnl::memory::format_tag wFormatMkl = dnnl::memory::format_tag::oihw;

  dnnl::memory::dims xDims = {bS, iC, iH, iW};
  dnnl::memory::dims wDims = {oC, iC, kH, kW};
  dnnl::memory::dims zDims = {bS, oC, oH, oW};

  // Use actual input data type - OneDNN supports f32, bf16, f16 for convolution
  auto type = getOneDnnDataType(input->dataType());

  std::vector<int> permut;
  if (0 == wFormat)
    permut = {3, 2, 0, 1};  // [kH, kW, iC, oC] -> [oC, iC, kH, kW]
  else if (2 == wFormat)
    permut = {0, 3, 1, 2};  // [oC, kH, kW, iC] -> [oC, iC, kH, kW]

  // memory descriptors for arrays

  // For NHWC arrays: the array has shape [bS, H, W, C] but OneDNN dims are in NCHW order [bS, C, H, W].
  // setBlockStrides assigns array strides to dims positionally, so we need a permutation that maps
  // NHWC array dim positions to NCHW OneDNN dim positions: permut[onednn_dim] = nhwc_array_dim
  // onednn[0]=bS <- nhwc[0]=bS, onednn[1]=C <- nhwc[3]=C, onednn[2]=H <- nhwc[1]=H, onednn[3]=W <- nhwc[2]=W
  const std::vector<int> nhwcPermut = {0, 3, 1, 2};

  sd_debug("Creating input descriptor\n",0);
  // input
  dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc x_user_md = dnnl::memory::desc(xDims, type, xzFormatMkl);
  if (!isNCHW)
    onednnUtils::setBlockStrides(*input, x_user_md, nhwcPermut);
  else
    onednnUtils::setBlockStrides(*input, x_user_md);

  sd_debug("Creating weight descriptor\n",0);

  // weights
  dnnl::memory::desc w_mkl_md = dnnl::memory::desc(wDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc w_user_md = dnnl::memory::desc(wDims, type, wFormatMkl);
  onednnUtils::setBlockStrides(*weights, w_user_md, permut);

  sd_debug("Creating bias descriptor\n",0);

  // bias
  dnnl::memory::desc b_mkl_md;
  if (bias != nullptr) b_mkl_md = dnnl::memory::desc({oC}, type, dnnl::memory::format_tag::x);

  sd_debug("Creating output\n",0);

  // output
  dnnl::memory::desc z_mkl_md = dnnl::memory::desc(zDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc z_user_md = dnnl::memory::desc(zDims, type, xzFormatMkl);
  if (!isNCHW)
    onednnUtils::setBlockStrides(*output, z_user_md, nhwcPermut);
  else
    onednnUtils::setBlockStrides(*output, z_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  sd_debug("Creating op descriptor\n",0);

  // operation primitive description (OneDNN 3.x API - no separate desc class)
  sd_debug("Creating prim  descriptor\n",0);

  dnnl::convolution_forward::primitive_desc op_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                          dnnl::algorithm::convolution_auto,
                                          x_mkl_md, w_mkl_md, b_mkl_md, z_mkl_md, strides, dilation, padding,
                                          padding_r);
  sd_debug("Created engine\n",0);

  // arguments (memory buffers) necessary for calculations
  std::unordered_map<int, dnnl::memory> args;

  dnnl::stream stream(engine);

  // provide memory buffers and check whether reorder is required

  // input
  onednnUtils::loadDataToMklStream(*input, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

  // weights
  onednnUtils::loadDataToMklStream(*weights, engine, stream, w_user_md, op_prim_desc.weights_desc(),
                                   args[DNNL_ARG_WEIGHTS]);

  // bias
  if (bias != nullptr) {
    auto b_mkl_mem = dnnl::memory(b_mkl_md, engine, const_cast<void *>(bias->buffer()));
    args[DNNL_ARG_BIAS] = b_mkl_mem;
  }

  // output
  auto z_user_mem =
      onednnUtils::loadDataToMklStream(*output, engine, stream, z_user_md, op_prim_desc.dst_desc(), args[DNNL_ARG_DST]);

  // run calculations
  dnnl::convolution_forward(op_prim_desc).execute(stream, args);

  // reorder outputs if necessary
  if (op_prim_desc.dst_desc() != z_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DST], z_user_mem).execute(stream, args[DNNL_ARG_DST], z_user_mem);

  stream.wait();
}

//////////////////////////////////////////////////////////////////////
static void conv2dBpMKLDNN(NDArray *input, NDArray *weights, NDArray *bias, NDArray *gradO,
                           NDArray *gradI, NDArray *gradW, NDArray *gradB, const int kH, const int kW, const int sH,
                           const int sW, const int pH, const int pW, const int dH, const int dW, const int paddingMode,
                           const int isNCHW, const int wFormat) {
  // mkl support weights/gradW in [oC, iC, kH, kW] format only

  sd::LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  sd::LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWoC, indWkH, indOoH);

  const int pWSame = (paddingMode == 2 && dW > 1) ? ((oW - 1) * sW + (kW - 1) * dW + 1 - iW) / 2
                                                  : pW;  // dH == 1 for causal mode in conv1d

  // Effective kernel sizes accounting for dilation (same fix as forward pass).
  const int eKH_bp = (kH - 1) * dH + 1;
  const int eKW_bp = (kW - 1) * dW + 1;

  dnnl::memory::dims strides = {sH, sW};
  dnnl::memory::dims padding = {pH, pW};
  dnnl::memory::dims padding_r = {(oH - 1) * sH - iH + eKH_bp - pH, (oW - 1) * sW - iW + eKW_bp - pWSame};
  dnnl::memory::dims dilation = {dH - 1, dW - 1};

  auto xzFormatMkl = isNCHW ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::nhwc;
  dnnl::memory::format_tag wFormatMkl = dnnl::memory::format_tag::oihw;

  dnnl::memory::dims xDims = {bS, iC, iH, iW};
  dnnl::memory::dims wDims = {oC, iC, kH, kW};
  dnnl::memory::dims zDims = {bS, oC, oH, oW};

  // Use actual input data type - OneDNN supports f32, bf16, f16 for convolution backward
  auto type = getOneDnnDataType(input->dataType());

  std::vector<int> permut;
  if (0 == wFormat)
    permut = {3, 2, 0, 1};  // [kH, kW, iC, oC] -> [oC, iC, kH, kW]
  else if (2 == wFormat)
    permut = {0, 3, 1, 2};  // [oC, kH, kW, iC] -> [oC, iC, kH, kW]

  // For NHWC arrays: the array has shape [bS, H, W, C] but OneDNN dims are in NCHW order [bS, C, H, W].
  // setBlockStrides assigns array strides to dims positionally, so we need a permutation that maps
  // NHWC array dim positions to NCHW OneDNN dim positions: permut[onednn_dim] = nhwc_array_dim
  const std::vector<int> nhwcPermut = {0, 3, 1, 2};

  // memory descriptors for arrays

  // input
  dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc x_user_md = dnnl::memory::desc(xDims, type, xzFormatMkl);
  if (!isNCHW)
    onednnUtils::setBlockStrides(*input, x_user_md, nhwcPermut);
  else
    onednnUtils::setBlockStrides(*input, x_user_md);

  // weights
  dnnl::memory::desc w_mkl_md = dnnl::memory::desc(wDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc w_user_md = dnnl::memory::desc(wDims, type, wFormatMkl);
  onednnUtils::setBlockStrides(*weights, w_user_md, permut);

  // gradO
  dnnl::memory::desc gradO_mkl_md = dnnl::memory::desc(zDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc gradO_user_md = dnnl::memory::desc(zDims, type, xzFormatMkl);
  if (!isNCHW)
    onednnUtils::setBlockStrides(*gradO, gradO_user_md, nhwcPermut);
  else
    onednnUtils::setBlockStrides(*gradO, gradO_user_md);

  // gradI
  dnnl::memory::desc gradI_mkl_md = dnnl::memory::desc(xDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc gradI_user_md = dnnl::memory::desc(xDims, type, xzFormatMkl);
  if (!isNCHW)
    onednnUtils::setBlockStrides(*gradI, gradI_user_md, nhwcPermut);
  else
    onednnUtils::setBlockStrides(*gradI, gradI_user_md);

  // gradW
  dnnl::memory::desc gradW_mkl_md = dnnl::memory::desc(wDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc gradW_user_md = dnnl::memory::desc(wDims, type, wFormatMkl);
  onednnUtils::setBlockStrides(*gradW, gradW_user_md, permut);

  // gradB
  dnnl::memory::desc gradB_mkl_md;
  if (gradB != nullptr) gradB_mkl_md = dnnl::memory::desc({oC}, type, dnnl::memory::format_tag::x);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // forward primitive description (OneDNN 3.x API)
  dnnl::convolution_forward::primitive_desc op_ff_prim_desc(engine, dnnl::prop_kind::forward_training,
                                             dnnl::algorithm::convolution_auto,
                                             x_mkl_md, w_mkl_md, gradB_mkl_md, gradO_mkl_md, strides, dilation, padding,
                                             padding_r);

  // backward data primitive description (OneDNN 3.x API)
  dnnl::convolution_backward_data::primitive_desc op_data_bp_prim_desc(engine, dnnl::algorithm::convolution_auto,
                                                        gradI_mkl_md, w_mkl_md,
                                                        gradO_mkl_md, strides, dilation, padding, padding_r,
                                                        op_ff_prim_desc);

  // backward weights primitive description (OneDNN 3.x API)
  dnnl::convolution_backward_weights::primitive_desc op_weights_bp_prim_desc(engine, dnnl::algorithm::convolution_auto,
                                                              x_mkl_md, gradW_mkl_md,
                                                              gradB_mkl_md, gradO_mkl_md, strides, dilation, padding,
                                                              padding_r, op_ff_prim_desc);

  // arguments (memory buffers) necessary for calculations
  std::unordered_map<int, dnnl::memory> args;

  dnnl::stream stream(engine);

  // provide memory buffers and check whether reorder is required

  // input
  onednnUtils::loadDataToMklStream(*input, engine, stream, x_user_md, op_weights_bp_prim_desc.src_desc(),
                                   args[DNNL_ARG_SRC]);

  // weights
  onednnUtils::loadDataToMklStream(*weights, engine, stream, w_user_md, op_data_bp_prim_desc.weights_desc(),
                                   args[DNNL_ARG_WEIGHTS]);

  // gradO
  auto gradO_user_mem = dnnl::memory(gradO_user_md, engine, const_cast<void *>(gradO->buffer()));
  const bool gradOReorderW = op_weights_bp_prim_desc.diff_dst_desc() != gradO_user_mem.get_desc();
  const bool gradOReorderD = op_data_bp_prim_desc.diff_dst_desc() != gradO_user_mem.get_desc();
  auto gradO_mkl_memW = gradOReorderW ? dnnl::memory(op_weights_bp_prim_desc.diff_dst_desc(), engine) : gradO_user_mem;
  auto gradO_mkl_memD = gradOReorderD ? dnnl::memory(op_data_bp_prim_desc.diff_dst_desc(), engine) : gradO_user_mem;
  if (gradOReorderW) dnnl::reorder(gradO_user_mem, gradO_mkl_memW).execute(stream, gradO_user_mem, gradO_mkl_memW);
  if (gradOReorderD) dnnl::reorder(gradO_user_mem, gradO_mkl_memD).execute(stream, gradO_user_mem, gradO_mkl_memD);
  args[DNNL_ARG_DIFF_DST] = gradO_mkl_memD;

  // gradI
  auto gradI_user_mem = onednnUtils::loadDataToMklStream(*gradI, engine, stream, gradI_user_md,
                                                         op_data_bp_prim_desc.diff_src_desc(), args[DNNL_ARG_DIFF_SRC]);

  // gradW
  auto gradW_user_mem = onednnUtils::loadDataToMklStream(
      *gradW, engine, stream, gradW_user_md, op_weights_bp_prim_desc.diff_weights_desc(), args[DNNL_ARG_DIFF_WEIGHTS]);

  // gradB
  if (gradB != nullptr) {
    auto gradB_mkl_mem = dnnl::memory(gradB_mkl_md, engine, gradB->buffer());
    args[DNNL_ARG_DIFF_BIAS] = gradB_mkl_mem;
  }

  // run backward data calculations
  dnnl::convolution_backward_data(op_data_bp_prim_desc).execute(stream, args);

  if (gradOReorderW || gradOReorderD) args[DNNL_ARG_DIFF_DST] = gradO_mkl_memW;

  // run backward weights calculations
  dnnl::convolution_backward_weights(op_weights_bp_prim_desc).execute(stream, args);

  // reorder gradI if necessary
  if (op_data_bp_prim_desc.diff_src_desc() != gradI_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DIFF_SRC], gradI_user_mem).execute(stream, args[DNNL_ARG_DIFF_SRC], gradI_user_mem);
  if (op_weights_bp_prim_desc.diff_weights_desc() != gradW_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DIFF_WEIGHTS], gradW_user_mem)
        .execute(stream, args[DNNL_ARG_DIFF_WEIGHTS], gradW_user_mem);

  stream.wait();

}



//////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv2d, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC]

  auto output = OUTPUT_VARIABLE(0);  // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

  sd::LongType sH = INT_ARG(2);                                                // strides height
  sd::LongType sW = INT_ARG(3);                                                // strides width
  sd::LongType pH = INT_ARG(4);                                                // paddings height
  sd::LongType pW = INT_ARG(5);                                                // paddings width
  sd::LongType dH = INT_ARG(6);                                                // dilations height
  sd::LongType dW = INT_ARG(7);                                                // dilations width
  int paddingMode = INT_ARG(8);                                       // 0-VALID, 1-SAME
  bool isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 0-NCHW,  1-NHWC
  int wFormat = block.getIArguments()->size() > 10
                    ? INT_ARG(10)
                    : 0;  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

  sd::LongType kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<sd::LongType>(weights->sizeAt(0));  // filter(kernel) height
  sd::LongType kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<sd::LongType>(weights->sizeAt(1));  // filter(kernel) width

  sd::LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  sd::LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWoC, indWkH, indOoH);

  ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, oC);
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CONV2D MKLDNN OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(
        bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
        "CONV2D MKLDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !",
        oC, bias->rankOf(), bias->lengthOf());

  conv2dMKLDNN(input, weights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW, wFormat);

  return sd::Status::OK;
}

PLATFORM_CHECK(conv2d, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  // Causal padding (paddingMode=2) produces asymmetric padding that oneDNN
  // cannot express correctly — the formula computes negative padding_r values
  // for dilated kernels. Route causal conv through the generic CPU path.
  int paddingMode = INT_ARG(8);

  // OneDNN does not support dilated convolutions with stride > 1 on any axis.
  // Attempting it yields dnnl_unimplemented (status 50). Route to the generic
  // CPU im2col path instead.
  sd::LongType sH = INT_ARG(2);
  sd::LongType sW = INT_ARG(3);
  sd::LongType dH = INT_ARG(6);
  sd::LongType dW = INT_ARG(7);
  bool dilatedWithStride = (dH > 1 || dW > 1) && (sH > 1 || sW > 1);

  // conv2d is available for float32, bfloat16, and float16 dtypes
  Requirements req("ONEDNN CONV2d OP");
  req.expectNotEq(makeInfoVariable(paddingMode, "paddingMode"),
                  makeInfoVariable(2, "CAUSAL")) &&
      req.expectFalse(makeInfoVariable(dilatedWithStride, "dilated+strided"),
                      "OneDNN does not support dilation>1 with stride>1") &&
      req.expectTrue(makeInfoVariable(isSupportedConv2dType(input->dataType()), TYPE_MSG_INPUT0),
                     "Must be FLOAT32, BFLOAT16, or HALF") &&
      req.expectTrue(makeInfoVariable(isSupportedConv2dType(weights->dataType()), TYPE_MSG_INPUT1),
                     "Must be FLOAT32, BFLOAT16, or HALF") &&
      req.expectTrue(makeInfoVariable(isSupportedConv2dType(output->dataType()), TYPE_MSG_OUTPUT),
                     "Must be FLOAT32, BFLOAT16, or HALF") &&
      req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1));
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv2d_bp, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;  // [oC]
  auto gradO = block.width() > 3
                   ? INPUT_VARIABLE(3)
                   : INPUT_VARIABLE(2);  // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

  auto gradI = OUTPUT_NULLIFIED(0);  // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
  auto gradW = OUTPUT_NULLIFIED(1);  // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto gradB = block.width() > 3 ? OUTPUT_NULLIFIED(2) : nullptr;  // [oC]

  sd::LongType kH = INT_ARG(0);                                               // filter(kernel) height
  sd::LongType kW = INT_ARG(1);                                               // filter(kernel) width
  sd::LongType sH = INT_ARG(2);                                               // strides height
  sd::LongType sW = INT_ARG(3);                                               // strides width
  sd::LongType pH = INT_ARG(4);                                               // paddings height
  sd::LongType pW = INT_ARG(5);                                               // paddings width
  sd::LongType dH = INT_ARG(6);                                               // dilations height
  sd::LongType dW = INT_ARG(7);                                               // dilations width
  int paddingMode = INT_ARG(8);                                      // 0-VALID, 1-SAME
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 0-NCHW, 1-NHWC
  int wFormat = block.getIArguments()->size() > 10
                    ? INT_ARG(10)
                    : 0;  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

  sd::LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  sd::LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWoC, indWkH, indOoH);

  sd::LongType trueoH, trueoW;  // true output height, width
  ConvolutionUtils::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, paddingMode);

  if (paddingMode)  // SAME
    ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

  std::vector<sd::LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, oC, trueoH, trueoW, 0, indIOioC, indOoH, indOoH + 1});
  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, oC);
  REQUIRE_TRUE(
      gradO->isSameShape(expectedGradOShape), 0,
      "CONV2D_BP MKLDNN OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CONV2D_BP MKLDNN OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                 "CONV2D_BP MKLDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, "
                 "%i instead !",
                 oC, bias->rankOf(), bias->lengthOf());

  conv2dBpMKLDNN(input, weights, bias, gradO, gradI, gradW, gradB, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW,
                 wFormat);

  return sd::Status::OK;
}

PLATFORM_CHECK(conv2d_bp, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, iC, oC] always
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;  // [oC]
  auto gradO = block.width() > 3
                   ? INPUT_VARIABLE(3)
                   : INPUT_VARIABLE(2);  // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
  auto gradW = OUTPUT_VARIABLE(1);  // [kH, kW, iC, oC] always
  auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;  // [oC]

  int paddingMode = INT_ARG(8);

  // OneDNN does not support dilated convolutions with stride > 1 on any axis.
  // Attempting it yields dnnl_unimplemented (status 50). Route to the generic
  // CPU im2col path instead.
  sd::LongType sH = INT_ARG(2);
  sd::LongType sW = INT_ARG(3);
  sd::LongType dH = INT_ARG(6);
  sd::LongType dW = INT_ARG(7);
  bool dilatedWithStride = (dH > 1 || dW > 1) && (sH > 1 || sW > 1);

  // conv2d_bp is available for float32, bfloat16, and float16 dtypes
  Requirements req("ONEDNN CONV2d_BP OP");
  req.expectNotEq(makeInfoVariable(paddingMode, "paddingMode"),
                  makeInfoVariable(2, "CAUSAL")) &&
      req.expectFalse(makeInfoVariable(dilatedWithStride, "dilated+strided"),
                      "OneDNN does not support dilation>1 with stride>1") &&
      req.expectTrue(makeInfoVariable(isSupportedConv2dType(input->dataType()), TYPE_MSG_INPUT0),
                     "Must be FLOAT32, BFLOAT16, or HALF") &&
      req.expectTrue(makeInfoVariable(isSupportedConv2dType(weights->dataType()), TYPE_MSG_INPUT1),
                     "Must be FLOAT32, BFLOAT16, or HALF") &&
      req.expectTrue(makeInfoVariable(isSupportedConv2dType(gradO->dataType()), TYPE_MSG_INPUT),
                     "Must be FLOAT32, BFLOAT16, or HALF") &&
      req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1));
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
