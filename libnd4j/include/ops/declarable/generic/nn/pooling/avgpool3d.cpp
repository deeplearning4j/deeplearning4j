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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 01.03.2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_avgpool3dnew)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/convolutions.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(avgpool3dnew, 1, 1, false, 0, 14) {
  auto input = INPUT_VARIABLE(0);     // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto output = OUTPUT_NULLIFIED(0);  // [bS, oD, oH, oW, iC] (NDHWC) or [bS, iC, oD, oH, oW] (NCDHW)

  LongType kD = INT_ARG(0);           // filter(kernel) depth
  LongType kH = INT_ARG(1);           // filter(kernel) height
  LongType kW = INT_ARG(2);           // filter(kernel) width
  LongType sD = INT_ARG(3);           // strides depth
  LongType sH = INT_ARG(4);           // strides height
  LongType sW = INT_ARG(5);           // strides width
  LongType pD = INT_ARG(6);           // paddings depth
  LongType pH = INT_ARG(7);           // paddings height
  LongType pW = INT_ARG(8);           // paddings width
  LongType dD = INT_ARG(9);           // dilations depth
  LongType dH = INT_ARG(10);          // dilations height
  LongType dW = INT_ARG(11);          // dilations width
  int isSameMode = INT_ARG(12);  // 1-SAME,  0-VALID
  int extraParam0 = INT_ARG(13);
  int isNCDHW = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1;  // 0-NCDHW, 1-NDHWC

  REQUIRE_TRUE(input->rankOf() == 5, 0, "AVGPOOL3DNEW OP: rank of input array must be equal to 5, but got %i instead !",
               input->rankOf());
  REQUIRE_TRUE(dD != 0 && dH != 0 && dW != 0, 0,
               "AVGPOOL3DNEW OP: dilation must not be zero, but got instead {%i, %i, %i}", dD, dH, dW);

  LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, 0, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC,
                                             indIOioD, indWiC, indWoC, indWkD);

  std::vector<LongType> expectedOutputShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, iC, oD, oH, oW, 0, indIOioC, indIOioD, indIOioD + 1, indIOioD + 2});
  REQUIRE_TRUE(output->isSameShape(expectedOutputShape), 0,
               "AVGPOOL3DNEW OP: wrong shape of output array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedOutputShape).c_str(), ShapeUtils::shapeAsString(output).c_str());

  if (!isNCDHW) {
    std::vector<sd::LongType> perm = {0, 4, 1, 2, 3};
    input = input->permute(perm, false, false);    // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
    output =output->permute(perm, false, false);  // [bS, oD, oH, oW, iC] -> [bS, iC, oD, oH, oW]
  }

  if (isSameMode)  // SAME
    ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

  // T extraParams[] = {};
  ConvolutionUtils::pooling3d(block, *input, *output, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, 1, extraParam0);

  if (!isNCDHW) {
     delete input;
     delete output;
  }

  return Status::OK;
}

DECLARE_TYPES(avgpool3dnew) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(avgpool3dnew) {
  LongType kD = INT_ARG(0);                                                  // filter(kernel) depth
  LongType kH = INT_ARG(1);                                                  // filter(kernel) height
  LongType kW = INT_ARG(2);                                                  // filter(kernel) width
  LongType sD = INT_ARG(3);                                                  // strides depth
  LongType sH = INT_ARG(4);                                                  // strides height
  LongType sW = INT_ARG(5);                                                  // strides width
  LongType pD = INT_ARG(6);                                                  // paddings depth
  LongType pH = INT_ARG(7);                                                  // paddings height
  LongType pW = INT_ARG(8);                                                  // paddings width
  LongType dD = INT_ARG(9);                                                  // dilations depth
  LongType dH = INT_ARG(10);                                                 // dilations height
  LongType dW = INT_ARG(11);                                                 // dilations width
  int isSameMode = INT_ARG(12);                                         // 1-SAME,  0-VALID
  int isNCDHW = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1;  // 0-NCDHW, 1-NDHWC

  REQUIRE_TRUE(dD != 0 && dH != 0 && dW != 0, 0,
               "AVGPOOL3DNEW op: dilation must not be zero, but got instead {%i, %i, %i}", dD, dH, dW);

  auto inputShapeInfo = inputShape->at(0);

  LongType idxID, idxIC;
  if (isNCDHW) {
    idxID = 2;
    idxIC = 1;
  } else {
    idxID = 1;
    idxIC = 4;
  }

  LongType bS = inputShapeInfo[1];          // batch size
  LongType iC = inputShapeInfo[idxIC + 1];  // input channels
  LongType iD = inputShapeInfo[idxID + 1];  // input depth
  LongType iH = inputShapeInfo[idxID + 2];  // input height
  LongType iW = inputShapeInfo[idxID + 3];  // input width

  LongType oD, oH, oW;  // output depth, height, width
  ConvolutionUtils::calcOutSizePool3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW,
                                      isSameMode);

  LongType outputShape[5];

  outputShape[0] = bS;

  if (isNCDHW) {
    outputShape[1] = iC;
    outputShape[2] = oD;
    outputShape[3] = oH;
    outputShape[4] = oW;
  } else {
    outputShape[1] = oD;
    outputShape[2] = oH;
    outputShape[3] = oW;
    outputShape[4] = iC;
  }
  // TF DOC: A Tensor. Has the same type as input.
  // TF DOC: A Tensor. Has the same type as input.
  auto ret = SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(ArrayOptions::dataType(inputShapeInfo),
                                                                             shape::order(inputShapeInfo),
                                                                             5,
                                                                             outputShape)->primary());
  return ret;
}

DECLARE_TYPES(avgpool3dnew_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(avgpool3dnew_bp, 2, 1, false, 0, 14) {
  auto input = INPUT_VARIABLE(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto gradO = INPUT_VARIABLE(1);    // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next
  auto gradI = OUTPUT_NULLIFIED(0);  // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon

  const LongType kD = INT_ARG(0);            // filter(kernel) depth
  const LongType kH = INT_ARG(1);            // filter(kernel) height
  const LongType kW = INT_ARG(2);            // filter(kernel) width
  const LongType sD = INT_ARG(3);            // strides depth
  const LongType sH = INT_ARG(4);            // strides height
  const LongType sW = INT_ARG(5);            // strides width
  LongType pD = INT_ARG(6);                  // paddings depth
  LongType pH = INT_ARG(7);                  // paddings height
  LongType pW = INT_ARG(8);                  // paddings width
  const LongType dD = INT_ARG(9);            // dilations depth
  const LongType dH = INT_ARG(10);           // dilations height
  const LongType dW = INT_ARG(11);           // dilations width
  const int isSameMode = INT_ARG(12);   // 1-SAME,  0-VALID
  const int extraParam0 = INT_ARG(13);  // define what divisor to use while averaging
  const int isNCDHW = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1;  // 0-NCDHW, 1-NDHWC

  REQUIRE_TRUE(input->rankOf() == 5, 0, "AVGPOOL3DNEW_BP op: input should have rank of 5, but got %i instead",
               input->rankOf());
  REQUIRE_TRUE(dD != 0 && dH != 0 && dW != 0, 0,
               "AVGPOOL3DNEW_BP op: dilation must not be zero, but got instead {%i, %i, %i}", dD, dH, dW);

  LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, 0, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC,
                                             indIOioD, indWiC, indWoC, indWkD);

  std::vector<LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, iC, oD, oH, oW, 0, indIOioC, indIOioD, indIOioD + 1, indIOioD + 2});
  std::vector<LongType> expectedGradIShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, iC, iD, iH, iW, 0, indIOioC, indIOioD, indIOioD + 1, indIOioD + 2});
  REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,
               "AVGPOOL3DNEW_BP op: wrong shape of output's gradients array (next epsilon), expected is %s, but got %s "
               "instead !",
               ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(
      gradI->isSameShape(expectedGradIShape), 0,
      "AVGPOOL3DNEW_BP op: wrong shape of input's gradients array (epsilon), expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(expectedGradIShape).c_str(), ShapeUtils::shapeAsString(gradI).c_str());

  if (!isNCDHW) {
    std::vector<sd::LongType> perm = {0, 4, 1, 2, 3};
    input = input->permute(perm, false, false);  // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
    gradI = gradI->permute(perm, false, false);  // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
    gradO =gradO->permute(perm, false, false);  // [bS, oD, oH, oW, iC] -> [bS, iC, oD, oH, oW]
  }

  if (isSameMode)  // SAME
    ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

  // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 -
  // poolingMode; 9 - divisor;
  ConvolutionUtils::pooling3dBP(block, *input, *gradO, *gradI, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, 1,
                                extraParam0);

  if (!isNCDHW) {
    delete input;
    delete gradI;
    delete gradO;
  }

  return Status::OK;
}

DECLARE_SHAPE_FN(avgpool3dnew_bp) {
  auto ret = SHAPELIST(ConstantShapeHelper::getInstance().castToDataType(inputShape->at(0),
                                                                         ArrayOptions::dataType(inputShape->at(1))));
  return ret;
}

}  // namespace ops
}  // namespace sd

#endif
