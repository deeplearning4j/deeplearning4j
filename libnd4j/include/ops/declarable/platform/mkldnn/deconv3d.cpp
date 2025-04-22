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
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <helpers/MKLDNNStream.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/helpers/convolutions.h>
#include <system/platform_boilerplate.h>

#include "mkldnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void deconv3dMKLDNN(NDArray* input, NDArray* weights, NDArray* bias, NDArray* output,
                           const sd::LongType kD, const sd::LongType kH, const sd::LongType kW, const sd::LongType sD, const sd::LongType sH, const sd::LongType sW,
                           const sd::LongType pD, const sd::LongType pH, const sd::LongType pW, const sd::LongType dD, const sd::LongType dH, const sd::LongType dW,
                           const bool isNCDHW, const int wFormat) {
  // mkl supports weights in [oC, iC, kD, kH, kW] only

  sd::LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  sd::LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWoC, indWiC, indWkD);

  dnnl::memory::dims strides = {sD, sH, sW};
  dnnl::memory::dims padding = {pD, pH, pW};
  dnnl::memory::dims padding_r = {(iD - 1) * sD - oD + kD - pD, (iH - 1) * sH - oH + kH - pH,
                                  (iW - 1) * sW - oW + kW - pW};
  dnnl::memory::dims dilation = {dD - 1, dH - 1, dW - 1};

  std::vector<int> permut;
  if (0 == wFormat)
    permut = {3, 4, 0, 1, 2};  // [kD, kH, kW, oC, iC] -> [oC, iC, kD, kH, kW]
  else if (1 == wFormat)
    permut = {1, 0, 2, 3, 4};  // [iC, oC, kD, kH, kW] -> [oC, iC, kD, kH, kW]
  else
    permut = {4, 0, 1, 2, 3};  // [iC, kD, kH, kW, oC] -> [oC, iC, kD, kH, kW]

  // input type
  dnnl::memory::data_type xType;
  if (input->dataType() == DataType::FLOAT32)
    xType = dnnl::memory::data_type::f32;
  else if (input->dataType() == DataType::HALF)
    xType = dnnl::memory::data_type::f16;
  else if (input->dataType() == DataType::UINT8)
    xType = dnnl::memory::data_type::u8;
  else
    xType = dnnl::memory::data_type::s8;

  // weights type
  dnnl::memory::data_type wType = xType;
  if (xType == dnnl::memory::data_type::u8) wType = dnnl::memory::data_type::s8;

  // output and bias type (have the same types)
  dnnl::memory::data_type zType;
  if (output->dataType() == DataType::FLOAT32)
    zType = dnnl::memory::data_type::f32;
  else if (output->dataType() == DataType::HALF)
    zType = dnnl::memory::data_type::f16;
  else if (output->dataType() == DataType::UINT8)
    zType = dnnl::memory::data_type::u8;
  else if (output->dataType() == DataType::INT8)
    zType = dnnl::memory::data_type::s8;
  else
    zType = dnnl::memory::data_type::s32;

  dnnl::memory::format_tag xFormatMkl = isNCDHW ? dnnl::memory::format_tag::ncdhw : dnnl::memory::format_tag::ndhwc;
  dnnl::memory::format_tag wFormatMkl = dnnl::memory::format_tag::oidhw;

  dnnl::memory::dims xDims = {bS, iC, iD, iH, iW};
  dnnl::memory::dims wDims = {oC, iC, kD, kH, kW};
  dnnl::memory::dims zDims = {bS, oC, oD, oH, oW};

  // memory descriptors for arrays

  // input
  dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xDims, xType, dnnl::memory::format_tag::any);
  dnnl::memory::desc x_user_md = dnnl::memory::desc(xDims, xType, xFormatMkl);
  onednnUtils::setBlockStrides(*input, x_user_md);

  // weights
  dnnl::memory::desc w_mkl_md = dnnl::memory::desc(wDims, wType, dnnl::memory::format_tag::any);
  dnnl::memory::desc w_user_md = dnnl::memory::desc(wDims, wType, wFormatMkl);
  onednnUtils::setBlockStrides(*weights, w_user_md, permut);

  // bias
  dnnl::memory::desc b_mkl_md;
  if (bias != nullptr) b_mkl_md = dnnl::memory::desc({oC}, zType, dnnl::memory::format_tag::x);

  // output
  dnnl::memory::desc z_mkl_md = dnnl::memory::desc(zDims, zType, dnnl::memory::format_tag::any);
  dnnl::memory::desc z_user_md = dnnl::memory::desc(zDims, zType, xFormatMkl);
  onednnUtils::setBlockStrides(*output, z_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // operation primitive description
  dnnl::deconvolution_forward::desc op_desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct,
                                            x_mkl_md, w_mkl_md, b_mkl_md, z_mkl_md, strides, dilation, padding,
                                            padding_r);
  dnnl::deconvolution_forward::primitive_desc op_prim_desc(op_desc, engine);

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
    auto b_mkl_mem = dnnl::memory(b_mkl_md, engine, const_cast<void*>(bias->buffer()));
    args[DNNL_ARG_BIAS] = b_mkl_mem;
  }

  // output
  auto z_user_mem =
      onednnUtils::loadDataToMklStream(*output, engine, stream, z_user_md, op_prim_desc.dst_desc(), args[DNNL_ARG_DST]);

  // run calculations
  dnnl::deconvolution_forward(op_prim_desc).execute(stream, args);

  // reorder outputs if necessary
  if (op_prim_desc.dst_desc() != z_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DST], z_user_mem).execute(stream, args[DNNL_ARG_DST], z_user_mem);

  stream.wait();

}

//////////////////////////////////////////////////////////////////////////
static void deconv3dBackPropMKLDNN(NDArray* input, NDArray* weights, NDArray* gradO, NDArray* gradI,
                                   NDArray* gradW, NDArray* gradB, const sd::LongType kD, const sd::LongType kH, const sd::LongType kW,
                                   const sd::LongType sD, const sd::LongType sH, const sd::LongType sW, const sd::LongType pD, const sd::LongType pH, const sd::LongType pW,
                                   const sd::LongType dD, const sd::LongType dH, const int dW, const bool isNCDHW, const int wFormat) {
  // mkl supports weights/gradW in [oC, iC, kD, kH, kW] format only

  sd::LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  sd::LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWoC, indWiC, indWkD);

  dnnl::memory::dims strides = {sD, sH, sW};
  dnnl::memory::dims padding = {pD, pH, pW};
  dnnl::memory::dims padding_r = {(iD - 1) * sD - oD + kD - pD, (iH - 1) * sH - oH + kH - pH,
                                  (iW - 1) * sW - oW + kW - pW};
  dnnl::memory::dims dilation = {dD - 1, dH - 1, dW - 1};

  std::vector<int> permut;
  if (0 == wFormat)
    permut = {3, 4, 0, 1, 2};  // [kD, kH, kW, oC, iC] -> [oC, iC, kD, kH, kW]
  else if (1 == wFormat)
    permut = {1, 0, 2, 3, 4};  // [iC, oC, kD, kH, kW] -> [oC, iC, kD, kH, kW]
  else
    permut = {4, 0, 1, 2, 3};  // [iC, kD, kH, kW, oC] -> [oC, iC, kD, kH, kW]

  // input type
  dnnl::memory::data_type xType =
      input->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;
  // weights type
  dnnl::memory::data_type wType =
      weights->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;
  // gradO type
  dnnl::memory::data_type gradOType =
      gradO->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;
  // gradI type
  dnnl::memory::data_type gradIType =
      gradI->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;
  // gradW type
  dnnl::memory::data_type gradWType =
      gradW->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;
  // gradB type
  dnnl::memory::data_type gradBType =
      gradB != nullptr
          ? (gradB->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16)
          : dnnl::memory::data_type::f32;

  dnnl::memory::format_tag xFormatMkl = isNCDHW ? dnnl::memory::format_tag::ncdhw : dnnl::memory::format_tag::ndhwc;
  dnnl::memory::format_tag wFormatMkl = dnnl::memory::format_tag::oidhw;

  dnnl::memory::dims xDims = {bS, iC, iD, iH, iW};
  dnnl::memory::dims wDims = {oC, iC, kD, kH, kW};
  dnnl::memory::dims zDims = {bS, oC, oD, oH, oW};

  // memory descriptors for arrays

  // input
  dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xDims, xType, dnnl::memory::format_tag::any);
  dnnl::memory::desc x_user_md = dnnl::memory::desc(xDims, xType, xFormatMkl);
  onednnUtils::setBlockStrides(*input, x_user_md);

  // weights
  dnnl::memory::desc w_mkl_md = dnnl::memory::desc(wDims, wType, dnnl::memory::format_tag::any);
  dnnl::memory::desc w_user_md = dnnl::memory::desc(wDims, wType, wFormatMkl);
  onednnUtils::setBlockStrides(*weights, w_user_md, permut);

  // gradO
  dnnl::memory::desc gradO_mkl_md = dnnl::memory::desc(zDims, gradOType, dnnl::memory::format_tag::any);
  dnnl::memory::desc gradO_user_md = dnnl::memory::desc(zDims, gradOType, xFormatMkl);
  onednnUtils::setBlockStrides(*gradO, gradO_user_md);

  // gradI
  dnnl::memory::desc gradI_mkl_md = dnnl::memory::desc(xDims, gradIType, dnnl::memory::format_tag::any);
  dnnl::memory::desc gradI_user_md = dnnl::memory::desc(xDims, gradIType, xFormatMkl);
  onednnUtils::setBlockStrides(*gradI, gradI_user_md);

  // gradW
  dnnl::memory::desc gradW_mkl_md = dnnl::memory::desc(wDims, gradWType, dnnl::memory::format_tag::any);
  dnnl::memory::desc gradW_user_md = dnnl::memory::desc(wDims, gradWType, wFormatMkl);
  onednnUtils::setBlockStrides(*gradW, gradW_user_md, permut);

  // gradB
  dnnl::memory::desc gradB_mkl_md;
  if (gradB != nullptr) gradB_mkl_md = dnnl::memory::desc({oC}, gradBType, dnnl::memory::format_tag::x);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // forward primitive description
  dnnl::deconvolution_forward::desc op_ff_desc(dnnl::prop_kind::forward_inference,
                                               dnnl::algorithm::deconvolution_direct, x_mkl_md, w_mkl_md, gradB_mkl_md,
                                               gradO_mkl_md, strides, dilation, padding, padding_r);
  dnnl::deconvolution_forward::primitive_desc op_ff_prim_desc(op_ff_desc, engine);

  // backward data primitive description
  dnnl::deconvolution_backward_data::desc op_data_bp_desc(dnnl::algorithm::deconvolution_direct, gradI_mkl_md, w_mkl_md,
                                                          gradO_mkl_md, strides, dilation, padding, padding_r);
  dnnl::deconvolution_backward_data::primitive_desc op_data_bp_prim_desc(op_data_bp_desc, engine, op_ff_prim_desc);

  // backward weights primitive description
  dnnl::deconvolution_backward_weights::desc op_weights_bp_desc(dnnl::algorithm::deconvolution_direct, x_mkl_md,
                                                                gradW_mkl_md, gradB_mkl_md, gradO_mkl_md, strides,
                                                                dilation, padding, padding_r);
  dnnl::deconvolution_backward_weights::primitive_desc op_weights_bp_prim_desc(op_weights_bp_desc, engine,
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
  auto gradO_user_mem = dnnl::memory(gradO_user_md, engine, const_cast<void*>(gradO->buffer()));
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
  dnnl::deconvolution_backward_data(op_data_bp_prim_desc).execute(stream, args);

  if (gradOReorderW || gradOReorderD) args[DNNL_ARG_DIFF_DST] = gradO_mkl_memW;

  // run backward weights calculations
  dnnl::deconvolution_backward_weights(op_weights_bp_prim_desc).execute(stream, args);

  // reorder gradI if necessary
  if (op_data_bp_prim_desc.diff_src_desc() != gradI_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DIFF_SRC], gradI_user_mem).execute(stream, args[DNNL_ARG_DIFF_SRC], gradI_user_mem);
  if (op_weights_bp_prim_desc.diff_weights_desc() != gradW_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DIFF_WEIGHTS], gradW_user_mem)
        .execute(stream, args[DNNL_ARG_DIFF_WEIGHTS], gradW_user_mem);

  stream.wait();

}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(deconv3d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto weights = INPUT_VARIABLE(1);  // [kD, kH, kW, oC, iC], [iC, oC, kD, kH, kW], [iC, kD, kH, kW, oC]
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC]

  auto output = OUTPUT_VARIABLE(0);  // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

  REQUIRE_TRUE(input->rankOf() == 5, 0,
               "CUSTOM DECONV3D_MKLDNN OP: rank of input array must be equal to 5, but got %i instead !",
               input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 5, 0,
               "CUSTOM DECONV3D_MKLDNN OP: rank of weights array must be equal to 5, but got %i instead !",
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
  int isSameMode = INT_ARG(12);                                                 // 0-SAME,  1-VALID
  int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;          // INT_ARG(13): 1-NDHWC, 0-NCDHW
  int wFormat = block.getIArguments()->size() > 14
                    ? INT_ARG(14)
                    : 0;  // 0 - [kD, kH, kW, oC, iC], 1 - [iC, oC, kD, kH, kW], 2 - [iC, kD, kH, kW, oC]

  sd::LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  sd::LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWoC, indWiC, indWkD);

  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, oC, iC);
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CUSTOM DECONV3D_MKLDNN OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                 "CUSTOM DECONV3D_MKLDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got "
                 "%i, %i instead !",
                 oC, bias->rankOf(), bias->lengthOf());

  if (isSameMode) {  // SAME
    // Note: we're intentionally swapping iH and oH, to calculated the padding for a"normal" conv (not deconv) forward
    // pass
    ConvolutionUtils::calcPadding3D(pD, pH, pW, iD, iH, iW, oD, oH, oW, kD, kH, kW, sD, sH, sW, dD, dH, dW);
  }

  deconv3dMKLDNN(input, weights, bias, output, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, isNCDHW, wFormat);

  return sd::Status::OK;
}

PLATFORM_CHECK(deconv3d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;

  auto output = INPUT_VARIABLE(0);

  sd::LongType dD = INT_ARG(9);           // dilations depth
  sd::LongType dH = INT_ARG(10);          // dilations height
  sd::LongType dW = INT_ARG(11);          // dilations width
  int isSameMode = INT_ARG(12);  // 0-SAME,  1-VALID

  Requirements req("ONEDNN DECONV3d OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectLessEq(makeInfoVariable(dD, "Dilation depth"), 1) &&
      req.expectLessEq(makeInfoVariable(dH, "Dilation height"), 1) &&
      req.expectLessEq(makeInfoVariable(dW, "Dilation width"), 1) &&
      req.expectFalse(makeInfoVariable(isSameMode, "isSameMode")) &&
      req.expectTrue(makeInfoVariable(
                         [input, weights, bias, output] {
                           const DataType xType = input->dataType();
                           const DataType wType = weights->dataType();
                           const DataType zType = output->dataType();
                           const DataType bType = bias != nullptr ? bias->dataType() : zType;
                           return (xType == DataType::FLOAT32 && wType == DataType::FLOAT32 &&
                                   bType == DataType::FLOAT32 && zType == DataType::FLOAT32) ||
                                  ((xType == DataType::UINT8 || xType == DataType::INT8) && wType == DataType::INT8 &&
                                   (zType == DataType::UINT8 || zType == DataType::INT8 || zType == DataType::INT32 ||
                                    zType == DataType::FLOAT32) &&
                                   bType == zType);
                         },
                         TYPECHECK_MSG),
                     NO_MSG);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(deconv3d_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto weights = INPUT_VARIABLE(1);  // [kD, kH, kW, oC, iC], [iC, oC, kD, kH, kW], [iC, kD, kH, kW, oC]
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;  // [oC]
  auto gradO = block.width() > 3
                   ? INPUT_VARIABLE(3)
                   : INPUT_VARIABLE(2);  // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), gradI
  auto gradW = OUTPUT_VARIABLE(1);  // [kD, kH, kW, oC, iC], [iC, oC, kD, kH, kW], [iC, kD, kH, kW, oC]
  auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;  // [oC]

  REQUIRE_TRUE(input->rankOf() == 5, 0,
               "CUSTOM DECONV3D_MKLDNN_BP OP: rank of input array must be equal to 5, but got %i instead !",
               input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 5, 0,
               "CUSTOM DECONV3D_MKLDNN_BP OP: rank of weights array must be equal to 5 , but got %i instead !",
               weights->rankOf());
  REQUIRE_TRUE(gradO->rankOf() == 5, 0,
               "CUSTOM DECONV3D_MKLDNN_BP OP: rank of output gradients (next epsilon) array must be equal to 5, but "
               "got %i instead !",
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
  int isSameMode = INT_ARG(12);                                                 // 0-SAME,  1-VALID
  int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;          // INT_ARG(13): 1-NDHWC, 0-NCDHW
  int wFormat = block.getIArguments()->size() > 14
                    ? INT_ARG(14)
                    : 0;  // 0 - [kD, kH, kW, oC, iC], 1 - [iC, oC, kD, kH, kW], 2 - [iC, kD, kH, kW, oC]

  sd::LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  sd::LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWoC, indWiC, indWkD);

  sd::LongType trueoD, trueoH, trueoW;  // true output height, width
  ConvolutionUtils::calcOutSizeDeconv3D(trueoD, trueoH, trueoW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH,
                                        iW, isSameMode);

  std::vector<sd::LongType> expectedGradOShape = ShapeUtils::composeShapeUsingDimsAndIdx(
      {bS, oC, trueoD, trueoH, trueoW, 0, indIOioC, indIOioD, indIOioD + 1, indIOioD + 2});
  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, oC, iC);
  REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,
               "CUSTOM DECONV3D_MKLDNN_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, "
               "but got %s instead !",
               ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CUSTOM DECONV3D_MKLDNN_BP OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                 "CUSTOM DECONV3D_MKLDNN_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but "
                 "got %i, %i instead !",
                 oC, bias->rankOf(), bias->lengthOf());

  if (isSameMode)  // Note: we're intentionally swapping iH and oH, to calculated the padding for a"normal" conv (not
                   // deconv) forward pass
    ConvolutionUtils::calcPadding3D(pD, pH, pW, iD, iH, iW, oD, oH, oW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

  deconv3dBackPropMKLDNN(input, weights, gradO, gradI, gradW, gradB, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW,
                         isNCDHW, wFormat);

  return sd::Status::OK;
}

PLATFORM_CHECK(deconv3d_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);    // [bS, iD, iH, iW, iC] (NHWC) or [bS, iD, iC, iH, iW] (NCDHW)
  auto weights = INPUT_VARIABLE(1);  // [kD, kH, kW, oC, iC], [iC, oC, kD, kH, kW], [iC, kD, kH, kW, oC]
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;  // [oC]
  auto gradO = block.width() > 3
                   ? INPUT_VARIABLE(3)
                   : INPUT_VARIABLE(2);  // [bS, oD, oH, oW, oC] (NHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iD, iH, iW, iC] (NHWC) or [bS, iC, iD, iH, iW] (NCDHW), gradI
  auto gradW = OUTPUT_VARIABLE(1);  // [kD, kH, kW, oC, iC], [iC, oC, kD, kH, kW], [iC, kD, kH, kW, oC]
  auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;  // [oC]

  int dD = INT_ARG(9);           // dilations depth
  int dH = INT_ARG(10);          // dilations height
  int dW = INT_ARG(11);          // dilations width
  int isSameMode = INT_ARG(12);  // 0-SAME,  1-VALID

  Requirements req("ONEDNN DECONV3d_BP OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectLessEq(makeInfoVariable(dD, "Dilation depth"), 1) &&
      req.expectLessEq(makeInfoVariable(dH, "Dilation height"), 1) &&
      req.expectLessEq(makeInfoVariable(dW, "Dilation width"), 1) &&
      req.expectFalse(makeInfoVariable(isSameMode, "isSameMode")) &&
      req.expectTrue(makeInfoVariable(
                         [input, weights, gradO, gradI, gradW, gradB] {
                           const DataType xType = input->dataType();
                           const DataType wType = weights->dataType();
                           const DataType gradOType = gradO->dataType();

                           const DataType gradIType = gradI->dataType();
                           const DataType gradWType = gradW->dataType();
                           const DataType gradBType = gradB != nullptr ? gradB->dataType() : DataType::FLOAT32;
                           return ((xType == DataType::FLOAT32 || xType == DataType::BFLOAT16) &&
                                   (wType == DataType::FLOAT32 || wType == DataType::BFLOAT16) &&
                                   (gradOType == DataType::FLOAT32 || gradOType == DataType::BFLOAT16) &&
                                   (gradIType == DataType::FLOAT32 || gradIType == DataType::BFLOAT16) &&
                                   (gradWType == DataType::FLOAT32 || gradWType == DataType::BFLOAT16) &&
                                   (gradBType == DataType::FLOAT32 || gradBType == DataType::BFLOAT16));
                         },
                         TYPECHECK_MSG),
                     NO_MSG);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
