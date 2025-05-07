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
static void conv3dMKLDNN(NDArray *input, NDArray *weights, NDArray *bias, NDArray *output,
                         const sd::LongType kD, const sd::LongType kH, const sd::LongType kW, const sd::LongType sD, const sd::LongType sH, const sd::LongType sW,
                         const sd::LongType pD, const sd::LongType pH, const sd::LongType pW, const sd::LongType dD, const sd::LongType dH, const sd::LongType dW,
                         const int paddingMode, const int isNCDHW, const int wFormat) {
  // mkl support weights  in [oC, iC, kD, kH, kW] format only

  sd::LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  sd::LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWiC, indWoC, indWkD);

  // const int pWSame = (paddingMode == 2 && dW > 1) ? ((oW - 1) * sW + (kW - 1) * dW + 1 - iW) / 2 : pW;       // dH ==
  // 1 for causal mode in conv1d

  dnnl::memory::dims strides = {sD, sH, sW};
  dnnl::memory::dims padding = {pD, pH, pW};
  // dnnl::memory::dims padding_r = { (oH - 1) * sH - iH + kH - pH, (oW - 1) * sW - iW + kW - pWSame };
  dnnl::memory::dims padding_r = {(oD - 1) * sD - iD + kD - pD, (oH - 1) * sH - iH + kH - pH,
                                  (oW - 1) * sW - iW + kW - pW};
  dnnl::memory::dims dilation = {dD - 1, dH - 1, dW - 1};

  auto xzFormatMkl = isNCDHW ? dnnl::memory::format_tag::ncdhw : dnnl::memory::format_tag::ndhwc;
  dnnl::memory::format_tag wFormatMkl = dnnl::memory::format_tag::oidhw;

  dnnl::memory::dims xDims = {bS, iC, iD, iH, iW};
  dnnl::memory::dims wDims = {oC, iC, kD, kH, kW};
  dnnl::memory::dims zDims = {bS, oC, oD, oH, oW};

  std::vector<int> permut;
  if (0 == wFormat)
    permut = {4, 3, 0, 1, 2};  // [kD, kH, kW, iC, oC] -> [oC, iC, kD, kH, kW]
  else if (2 == wFormat)
    permut = {0, 4, 1, 2, 3};  // [oC, kD, kH, kW, iC] -> [oC, iC, kD, kH, kW]

  auto type = dnnl::memory::data_type::f32;

  // memory descriptors for arrays

  // input
  dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc x_user_md = dnnl::memory::desc(xDims, type, xzFormatMkl);
  onednnUtils::setBlockStrides(*input, x_user_md);

  // weights
  dnnl::memory::desc w_mkl_md = dnnl::memory::desc(wDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc w_user_md = dnnl::memory::desc(wDims, type, wFormatMkl);
  onednnUtils::setBlockStrides(*weights, w_user_md, permut);

  // bias
  dnnl::memory::desc b_mkl_md;
  if (bias != nullptr) b_mkl_md = dnnl::memory::desc({oC}, type, dnnl::memory::format_tag::x);

  // output
  dnnl::memory::desc z_mkl_md = dnnl::memory::desc(zDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc z_user_md = dnnl::memory::desc(zDims, type, xzFormatMkl);
  onednnUtils::setBlockStrides(*output, z_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // operation primitive description
  dnnl::convolution_forward::desc op_desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_auto,
                                          x_mkl_md, w_mkl_md, b_mkl_md, z_mkl_md, strides, dilation, padding,
                                          padding_r);
  dnnl::convolution_forward::primitive_desc op_prim_desc(op_desc, engine);

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
static void conv3dBpMKLDNN(NDArray *input, NDArray *weights, NDArray *bias, NDArray *gradO,
                           NDArray *gradI, NDArray *gradW, NDArray *gradB, const sd::LongType kD, const sd::LongType kH, const sd::LongType kW,
                           const sd::LongType sD, const sd::LongType sH, const sd::LongType sW, const sd::LongType pD, const sd::LongType pH, const sd::LongType pW,
                           const sd::LongType dD, const sd::LongType dH, const sd::LongType dW, const int paddingMode, const int isNCDHW,
                           const int wFormat) {
  // mkl support weights/gradW in [oC, iC, kD, kH, kW] format only

  sd::LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  sd::LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWiC, indWoC, indWkD);

  // const int pWSame = (paddingMode == 2 && dW > 1) ? ((oW - 1) * sW + (kW - 1) * dW + 1 - iW) / 2 : pW;       // dH ==
  // 1 for causal mode in conv1d

  dnnl::memory::dims strides = {sD, sH, sW};
  dnnl::memory::dims padding = {pD, pH, pW};
  // dnnl::memory::dims padding_r = { (oH - 1) * sH - iH + kH - pH, (oW - 1) * sW - iW + kW - pWSame };
  dnnl::memory::dims padding_r = {(oD - 1) * sD - iD + kD - pD, (oH - 1) * sH - iH + kH - pH,
                                  (oW - 1) * sW - iW + kW - pW};
  dnnl::memory::dims dilation = {dD - 1, dH - 1, dW - 1};

  auto xzFormatMkl = isNCDHW ? dnnl::memory::format_tag::ncdhw : dnnl::memory::format_tag::ndhwc;
  dnnl::memory::format_tag wFormatMkl = dnnl::memory::format_tag::oidhw;

  dnnl::memory::dims xDims = {bS, iC, iD, iH, iW};
  dnnl::memory::dims wDims = {oC, iC, kD, kH, kW};
  dnnl::memory::dims zDims = {bS, oC, oD, oH, oW};

  auto type = dnnl::memory::data_type::f32;

  std::vector<int> permut;
  if (0 == wFormat)
    permut = {4, 3, 0, 1, 2};  // [kD, kH, kW, iC, oC] -> [oC, iC, kD, kH, kW]
  else if (2 == wFormat)
    permut = {0, 4, 1, 2, 3};  // [oC, kD, kH, kW, iC] -> [oC, iC, kD, kH, kW]

  // memory descriptors for arrays

  // input
  dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc x_user_md = dnnl::memory::desc(xDims, type, xzFormatMkl);
  onednnUtils::setBlockStrides(*input, x_user_md);

  // weights
  dnnl::memory::desc w_mkl_md = dnnl::memory::desc(wDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc w_user_md = dnnl::memory::desc(wDims, type, wFormatMkl);
  onednnUtils::setBlockStrides(*weights, w_user_md, permut);

  // gradO
  dnnl::memory::desc gradO_mkl_md = dnnl::memory::desc(zDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc gradO_user_md = dnnl::memory::desc(zDims, type, xzFormatMkl);

  onednnUtils::setBlockStrides(*gradO, gradO_user_md);

  // gradI
  dnnl::memory::desc gradI_mkl_md = dnnl::memory::desc(xDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc gradI_user_md = dnnl::memory::desc(xDims, type, xzFormatMkl);

  onednnUtils::setBlockStrides(*gradI, gradI_user_md);

  // gradW
  dnnl::memory::desc gradW_mkl_md = dnnl::memory::desc(wDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc gradW_user_md = dnnl::memory::desc(wDims, type, wFormatMkl);
  onednnUtils::setBlockStrides(*gradW, gradW_user_md, permut);

  // gradB
  dnnl::memory::desc gradB_mkl_md;
  if (gradB != nullptr) gradB_mkl_md = dnnl::memory::desc({oC}, type, dnnl::memory::format_tag::x);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // forward primitive description
  dnnl::convolution_forward::desc op_ff_desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_auto,
                                             x_mkl_md, w_mkl_md, gradB_mkl_md, gradO_mkl_md, strides, dilation, padding,
                                             padding_r);
  dnnl::convolution_forward::primitive_desc op_ff_prim_desc(op_ff_desc, engine);

  // backward data primitive description
  dnnl::convolution_backward_data::desc op_data_bp_desc(dnnl::algorithm::convolution_auto, gradI_mkl_md, w_mkl_md,
                                                        gradO_mkl_md, strides, dilation, padding, padding_r);
  dnnl::convolution_backward_data::primitive_desc op_data_bp_prim_desc(op_data_bp_desc, engine, op_ff_prim_desc);

  // backward weights primitive description
  dnnl::convolution_backward_weights::desc op_weights_bp_desc(dnnl::algorithm::convolution_auto, x_mkl_md, gradW_mkl_md,
                                                              gradB_mkl_md, gradO_mkl_md, strides, dilation, padding,
                                                              padding_r);
  dnnl::convolution_backward_weights::primitive_desc op_weights_bp_prim_desc(op_weights_bp_desc, engine,
                                                                             op_ff_prim_desc);

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
PLATFORM_IMPL(conv3dnew, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto weights = INPUT_VARIABLE(1);  // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC]
  auto output = OUTPUT_VARIABLE(0);  // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

  REQUIRE_TRUE(input->rankOf() == 5, 0,
               "CUSTOM CONV3D MKLDNN OP: rank of input array must be equal to 5, but got %i instead !",
               input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 5, 0,
               "CUSTOM CONV3D MKLDNN OP: rank of weights array must be equal to 5, but got %i instead !",
               weights->rankOf());

  sd::LongType kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<sd::LongType>(weights->sizeAt(0));  // filter(kernel) depth
  sd::LongType kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<sd::LongType>(weights->sizeAt(1));  // filter(kernel) height
  sd::LongType kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<sd::LongType>(weights->sizeAt(2));  // filter(kernel) width
  sd::LongType sD = INT_ARG(3);                                                          // strides depth
  sd::LongType sH = INT_ARG(4);                                                          // strides height
  sd::LongType sW = INT_ARG(5);                                                          // strides width
  sd::LongType pD = INT_ARG(6);                                                          // paddings depth
  sd::LongType pH = INT_ARG(7);                                                          // paddings height
  sd::LongType pW = INT_ARG(8);                                                          // paddings width
  sd::LongType dD = INT_ARG(9);                                                          // dilations depth
  sd::LongType dH = INT_ARG(10);                                                         // dilations height
  sd::LongType dW = INT_ARG(11);                                                         // dilations width
  int paddingMode = INT_ARG(12);                                                // 0-SAME,  1-VALID
  int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;          // INT_ARG(13): 1-NDHWC, 0-NCDHW
  int wFormat = block.getIArguments()->size() > 14
                    ? INT_ARG(14)
                    : 0;  // 0 - [kD, kH, kW, iC, oC], 1 - [oC, iC, kD, kH, kW], 2 - [oC, kD, kH, kW, iC]

  sd::LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  sd::LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWiC, indWoC, indWkD);

  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, iC, oC);
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CUSTOM CONV3D MKLDNN OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                 "CUSTOM CONV3D MKLDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got "
                 "%i, %i instead !",
                 oC, bias->rankOf(), bias->lengthOf());

  if (paddingMode)  // SAME
    ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

  conv3dMKLDNN(input, weights, bias, output, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, paddingMode, isNCDHW,
               wFormat);

  return sd::Status::OK;
}

PLATFORM_CHECK(conv3dnew, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto weights = INPUT_VARIABLE(1);  // [kD, kH, kW, iC, oC] always
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC]
  auto output = OUTPUT_VARIABLE(0);  // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)
  Requirements req("ONEDNN CONV3d OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectTrue(sd::ONEDNNStream::isSupported({input, weights, bias, output}), ONEDNN_STREAM_NOT_SUPPORTED);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv3dnew_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto weights = INPUT_VARIABLE(1);  // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;  // [oC]
  auto gradO = block.width() > 3
                   ? INPUT_VARIABLE(3)
                   : INPUT_VARIABLE(2);  // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

  auto gradI = OUTPUT_NULLIFIED(0);  // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon
  auto gradW = OUTPUT_NULLIFIED(1);  // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
  auto gradB = block.width() > 3 ? OUTPUT_NULLIFIED(2) : nullptr;  // [oC]

  REQUIRE_TRUE(input->rankOf() == 5, 0,
               "CUSTOM CONV3D_BP MKLDNN OP: rank of input array must be equal to 5, but got %i instead !",
               input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 5, 0,
               "CUSTOM CONV3D_BP MKLDNN OP: rank of weights array must be equal to 5, but got %i instead !",
               weights->rankOf());
  REQUIRE_TRUE(gradO->rankOf() == 5, 0,
               "CUSTOM CONV3D_BP MKLDNN OP: rank of output gradients (next epsilon) array must be equal to 5, but got "
               "%i instead !",
               gradO->rankOf());

  sd::LongType kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<sd::LongType>(weights->sizeAt(0));  // filter(kernel) depth
  sd::LongType kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<sd::LongType>(weights->sizeAt(1));  // filter(kernel) height
  sd::LongType kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<sd::LongType>(weights->sizeAt(2));  // filter(kernel) width
  sd::LongType sD = INT_ARG(3);                                                          // strides depth
  sd::LongType sH = INT_ARG(4);                                                          // strides height
  sd::LongType sW = INT_ARG(5);                                                          // strides width
  sd::LongType pD = INT_ARG(6);                                                          // paddings depth
  sd::LongType pH = INT_ARG(7);                                                          // paddings height
  sd::LongType pW = INT_ARG(8);                                                          // paddings width
  sd::LongType dD = INT_ARG(9);                                                          // dilations depth
  sd::LongType dH = INT_ARG(10);                                                         // dilations height
  sd::LongType dW = INT_ARG(11);                                                         // dilations width
  int paddingMode = INT_ARG(12);                                                // 1-SAME,  0-VALID
  int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;          // INT_ARG(13): 1-NDHWC, 0-NCDHW
  int wFormat = block.getIArguments()->size() > 14
                    ? INT_ARG(14)
                    : 0;  // 0 - [kD, kH, kW, iC, oC], 1 - [oC, iC, kD, kH, kW], 2 - [oC, kD, kH, kW, iC]

  sd::LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  sd::LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWiC, indWoC, indWkD);

  if (paddingMode)  // SAME
    ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

  sd::LongType trueoD, trueoH, trueoW;  // true output depth/height/width
  ConvolutionUtils::calcOutSizePool3D(trueoD, trueoH, trueoW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH,
                                      iW, paddingMode);

  std::vector<sd::LongType> expectedGradOShape = ShapeUtils::composeShapeUsingDimsAndIdx(
      {bS, oC, trueoD, trueoH, trueoW, 0, indIOioC, indIOioD, indIOioD + 1, indIOioD + 2});
  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, iC, oC);
  REQUIRE_TRUE(
      gradO->isSameShape(expectedGradOShape), 0,
      "CUSTOM CONV3D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CUSTOM CONV3D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                 "CUSTOM CONV3D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, "
                 "%i instead !",
                 oC, bias->rankOf(), bias->lengthOf());

  conv3dBpMKLDNN(input, weights, bias, gradO, gradI, gradW, gradB, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW,
                 paddingMode, isNCDHW, wFormat);

  return sd::Status::OK;
}

PLATFORM_CHECK(conv3dnew_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto weights = INPUT_VARIABLE(1);  // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;  // [oC]
  auto gradO = block.width() > 3
                   ? INPUT_VARIABLE(3)
                   : INPUT_VARIABLE(2);  // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon
  auto gradW = OUTPUT_VARIABLE(1);  // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
  auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;  // [oC]

  Requirements req("ONEDNN CONV3d_BP OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectTrue(sd::ONEDNNStream::isSupported({input, weights, bias, gradO, gradI, gradW, gradB}),
                     ONEDNN_STREAM_NOT_SUPPORTED);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
