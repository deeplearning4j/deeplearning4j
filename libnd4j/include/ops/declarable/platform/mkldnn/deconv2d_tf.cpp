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
static void deconv2TFdBpMKLDNN(const NDArray* weights, const NDArray* gradO, NDArray* gradI, const sd::LongType bS, const sd::LongType iC,
                               const sd::LongType iH, const sd::LongType iW, const sd::LongType oC, const sd::LongType oH, const sd::LongType oW, const sd::LongType kH,
                               const sd::LongType kW, const sd::LongType sH, const sd::LongType sW, const sd::LongType pH, const sd::LongType pW, const sd::LongType dH,
                               const sd::LongType dW, const bool isNCHW, const int wFormat) {
  // gradI [bS, iH, iW, iC], mkl doesn't support ndhwc format
  // weights [oC, iC, kH, kW] always, mkl doesn't support weights format [kH, kW, iC, oC]
  // gradO [bS, oH, oW, oC]

  dnnl::memory::dims strides = {sH, sW};
  dnnl::memory::dims dilation = {dH - 1, dW - 1};
  dnnl::memory::dims padding = {pH, pW};
  dnnl::memory::dims padding_r = {(oH - 1) * sH - iH + kH - pH, (oW - 1) * sW - iW + kW - pW};

  // weights type
  dnnl::memory::data_type wType =
      weights->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;
  // gradO type
  dnnl::memory::data_type gradOType =
      gradO->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;
  // gradI type
  dnnl::memory::data_type gradIType =
      gradI->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;

  dnnl::memory::format_tag xFormatMkl = isNCHW ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::nhwc;
  dnnl::memory::format_tag wFormatMkl = dnnl::memory::format_tag::oihw;

  dnnl::memory::dims xDims = {bS, iC, iH, iW};
  dnnl::memory::dims wDims = {oC, iC, kH, kW};
  dnnl::memory::dims zDims = {bS, oC, oH, oW};

  // memory descriptors for arrays

  // input
  dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xDims, gradOType, dnnl::memory::format_tag::any);

  // weights
  dnnl::memory::desc w_mkl_md = dnnl::memory::desc(wDims, wType, dnnl::memory::format_tag::any);
  dnnl::memory::desc w_user_md = dnnl::memory::desc(wDims, wType, wFormatMkl);
  onednnUtils::setBlockStrides(*weights, w_user_md, {3, 2, 0, 1});  // permute [kH, kW, iC, oC] -> [oC, iC, kH, kW]

  // gradO
  dnnl::memory::desc gradO_mkl_md = dnnl::memory::desc(zDims, gradOType, dnnl::memory::format_tag::any);
  dnnl::memory::desc gradO_user_md = dnnl::memory::desc(zDims, gradOType, xFormatMkl);
  onednnUtils::setBlockStrides(*gradO, gradO_user_md);

  // gradI
  dnnl::memory::desc gradI_mkl_md = dnnl::memory::desc(xDims, gradIType, dnnl::memory::format_tag::any);
  dnnl::memory::desc gradI_user_md = dnnl::memory::desc(xDims, gradIType, xFormatMkl);
  onednnUtils::setBlockStrides(*gradI, gradI_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // forward primitive description
  dnnl::convolution_forward::desc op_ff_desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_auto,
                                             x_mkl_md, w_mkl_md, gradO_mkl_md, strides, dilation, padding, padding_r);
  dnnl::convolution_forward::primitive_desc op_ff_prim_desc(op_ff_desc, engine);

  // backward data primitive description
  dnnl::convolution_backward_data::desc op_data_bp_desc(dnnl::algorithm::convolution_auto, gradI_mkl_md, w_mkl_md,
                                                        gradO_mkl_md, strides, dilation, padding, padding_r);
  dnnl::convolution_backward_data::primitive_desc op_data_bp_prim_desc(op_data_bp_desc, engine, op_ff_prim_desc);

  // arguments (memory buffers) necessary for calculations
  std::unordered_map<sd::LongType, dnnl::memory> args;

  dnnl::stream stream(engine);

  // provide memory buffers and check whether reorder is required

  // weights
  onednnUtils::loadDataToMklStream(*weights, engine, stream, w_user_md, op_data_bp_prim_desc.weights_desc(),
                                   args[DNNL_ARG_WEIGHTS]);

  // gradO
  onednnUtils::loadDataToMklStream(*gradO, engine, stream, gradO_user_md, op_data_bp_prim_desc.diff_dst_desc(),
                                   args[DNNL_ARG_DIFF_DST]);

  // gradI
  auto gradI_user_mem = onednnUtils::loadDataToMklStream(*gradI, engine, stream, gradI_user_md,
                                                         op_data_bp_prim_desc.diff_src_desc(), args[DNNL_ARG_DIFF_SRC]);

  // run backward data calculations
  dnnl::convolution_backward_data(op_data_bp_prim_desc).execute(stream, args);

  // reorder gradI if necessary
  if (op_data_bp_prim_desc.diff_src_desc() != gradI_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DIFF_SRC], gradI_user_mem).execute(stream, args[DNNL_ARG_DIFF_SRC], gradI_user_mem);

  stream.wait();

  // shape::printArray(z_mkl_mem.map_data<float>(),8);
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(deconv2d_tf, ENGINE_CPU) {
  auto gradO = INPUT_VARIABLE(2);       // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
  auto weights = INPUT_VARIABLE(1);     // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto gradIShape = INPUT_VARIABLE(0);  // [4] - shape of input of conv2d (that is shape of gradI)

  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon

  sd::LongType kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<sd::LongType>(weights->sizeAt(0));  // filter(kernel) height
  sd::LongType kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<sd::LongType>(weights->sizeAt(1));  // filter(kernel) width
  sd::LongType sH = INT_ARG(2);                                                          // strides height
  sd::LongType sW = INT_ARG(3);                                                          // strides width
  sd::LongType pH = INT_ARG(4);                                                          // paddings height
  sd::LongType pW = INT_ARG(5);                                                          // paddings width
  sd::LongType dH = INT_ARG(6);                                                          // dilations height
  sd::LongType dW = INT_ARG(7);                                                          // dilations width
  int isSameMode = INT_ARG(8);                                                  // 0-VALID, 1-SAME
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;             // INT_ARG(9): 1-NHWC, 0-NCHW
  int wFormat = block.getIArguments()->size() > 10
                    ? INT_ARG(10)
                    : 0;  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

  const sd::LongType rank = gradO->rankOf();

  REQUIRE_TRUE(weights->rankOf() == rank, 0,
               "CUSTOM DECONV2D_TF MKLDNN OP: rank of weights array must be equal to 4, but got %i instead !",
               weights->rankOf());
  REQUIRE_TRUE(gradIShape->rankOf() == 1, 0,
               "CUSTOM DECONV2D_TF MKLDNN OP: rank of array with output shape must be equal to 1, but got %i instead !",
               gradIShape->rankOf());
  REQUIRE_TRUE(
      gradIShape->lengthOf() == rank, 0,
      "CUSTOM DECONV2D_TF MKLDNN OP: length of array with output shape must be equal to 4, but got %i instead !",
      gradIShape->lengthOf());

  int indIOioC, indIiH, indWoC(3), indOoH;
  if (!isNCHW) {
    indIOioC = 3;
    indIiH = 1;
    indOoH = 1;
  } else {
    indIOioC = 1;
    indIiH = 2;
    indOoH = 2;
  }

  std::vector<sd::LongType> gradIShapeVector = gradIShape->template asVectorT<sd::LongType>();

  const sd::LongType bS = gradIShapeVector[0];           // batch size
  const sd::LongType iH = gradIShapeVector[indIiH];      // input height
  const sd::LongType iW = gradIShapeVector[indIiH + 1];  // input width
  const sd::LongType iC = gradIShapeVector[indIOioC];    // input channels
  const sd::LongType oC = weights->sizeAt(indWoC);       // output channels
  const sd::LongType oH = gradO->sizeAt(indOoH);         // input height
  const sd::LongType oW = gradO->sizeAt(indOoH);         // input width

  sd::LongType trueoH, trueoW;  // true output height, width
  ConvolutionUtils::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

  std::vector<sd::LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, oC, trueoH, trueoW, 0, indIOioC, indOoH, indOoH + 1});
  std::vector<sd::LongType> expectedWeightsShape = {kH, kW, iC, oC};
  REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,
               "CUSTOM DECONV2D_TF MKLDNN OP: wrong shape of input array, basing on array with output shape expected "
               "is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CUSTOM DECONV2D_TF MKLDNN OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());

  if (isSameMode)  // SAME
    ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

  // // mkl supports only [oC, iC, kH, kW] for weights


  deconv2TFdBpMKLDNN(weights, gradO, gradI, bS, iC, iH, iW, oC, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, isNCHW,
                     wFormat);

  return sd::Status::OK;
}

PLATFORM_CHECK(deconv2d_tf, ENGINE_CPU) {
  auto weights = INPUT_VARIABLE(1);  // [kH, kW, iC, oC] always
  auto gradO = INPUT_VARIABLE(2);    // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next
  auto gradI = OUTPUT_VARIABLE(0);   // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCDHW), gradI
  Requirements req("ONEDNN DECONV2d_TF OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectTrue(makeInfoVariable(
                         [weights, gradI, gradO] {
                           const DataType wType = weights->dataType();
                           const DataType gradOType = gradO->dataType();
                           const DataType gradIType = gradI->dataType();
                           return ((wType == DataType::FLOAT32 || wType == DataType::BFLOAT16) &&
                                   (gradOType == DataType::FLOAT32 || gradOType == DataType::BFLOAT16) &&
                                   (gradIType == DataType::FLOAT32 || gradIType == DataType::BFLOAT16));
                         },
                         TYPECHECK_MSG),
                     NO_MSG);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
