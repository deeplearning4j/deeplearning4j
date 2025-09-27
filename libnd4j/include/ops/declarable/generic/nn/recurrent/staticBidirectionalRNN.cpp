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
// @author Yurii Shyrma, created on 03.04.2018
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reverse.h>
#include <ops/declarable/helpers/rnn.h>
#include <ops/declarable/helpers/transforms.h>

#if NOT_EXCLUDED(OP_static_bi_directional_rnn)
namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(static_bidirectional_rnn, 7, 3, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);     // input [time x bS x inSize]
  auto WxFW = INPUT_VARIABLE(1);  // input-to-hidden  weights for forward  RNN, [inSize  x numUnitsFW]
  auto WhFW = INPUT_VARIABLE(2);  // hidden-to-hidden weights for forward  RNN, [numUnitsFW x numUnitsFW]
  auto bFW = INPUT_VARIABLE(3);   // biases for forward  RNN, [2*numUnitsFW]
  auto WxBW = INPUT_VARIABLE(4);  // input-to-hidden  weights for backward RNN, [inSize  x numUnitsBW]
  auto WhBW = INPUT_VARIABLE(5);  // hidden-to-hidden weights for backward RNN, [numUnitsBW x numUnitsBW]
  auto bBW = INPUT_VARIABLE(6);   // biases for backward RNN, [2*v]

  NDArray* h0FW = nullptr;  // initial cell output for forward  RNN (at time step = 0) [bS x numUnitsFW]
  NDArray* h0BW = nullptr;  // initial cell output for backward RNN (at time step = 0) [bS x numUnitsBW]
  NDArray* maxTimeStep =
      nullptr;  // vector [bS] containing integer values within [0,time), each element of this vector set max time step
  // per each input in batch, this means there are no calculations for time >= maxTimeStep

  switch (block.width()) {
    case 8:
      maxTimeStep = INPUT_VARIABLE(7);
      break;
    case 9:
      h0FW = INPUT_VARIABLE(7);
      h0BW = INPUT_VARIABLE(8);
      break;
    case 10:
      h0FW = INPUT_VARIABLE(7);
      h0BW = INPUT_VARIABLE(8);
      maxTimeStep = INPUT_VARIABLE(9);
      break;
  }

  auto h = OUTPUT_VARIABLE(0);  // cell outputs [time x bS x (numUnitsFW + numUnitsBW)], that is per each time step
  auto hFWFinal = OUTPUT_VARIABLE(1);  // final cell out for forward  RNN [bS x numUnitsFW]
  auto hBWFinal = OUTPUT_VARIABLE(2);  // final cell out for backward RNN [bS x numUnitsBF]

  REQUIRE_TRUE(x->rankOf() == 3, 0,
               "STATIC_BIDIRECTIONAL_RNN custom operation: input array must have rank = 3, but got %i instead !",
               x->rankOf());
  REQUIRE_TRUE(WxFW->rankOf() == 2, 0,
               "STATIC_BIDIRECTIONAL_RNN custom operation: input-to-hidden weights array (for forward  RNN) must have "
               "rank = 2, but got %i instead !",
               WxFW->rankOf());
  REQUIRE_TRUE(WxBW->rankOf() == 2, 0,
               "STATIC_BIDIRECTIONAL_RNN custom operation: input-to-hidden weights array (for backward RNN) must have "
               "rank = 2, but got %i instead !",
               WxBW->rankOf());

  const LongType inRank = x->rankOf();
  const LongType time = x->sizeAt(0);
  const LongType bS = x->sizeAt(1);
  const LongType numUnitsFW = WxFW->sizeAt(1);
  const LongType numUnitsBW = WxBW->sizeAt(1);

  const std::vector<LongType> expectedWhFWshape = {numUnitsFW, numUnitsFW};
  const std::vector<LongType> expectedWhBWshape = {numUnitsBW, numUnitsBW};
  const std::vector<LongType> expectedbFWshape = {2 * numUnitsFW};
  const std::vector<LongType> expectedbBWshape = {2 * numUnitsBW};

  REQUIRE_TRUE(WhFW->isSameShape(expectedWhFWshape), 0,
               "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of hidden-to-hidden weights array (for forward  "
               "RNN), expected is %s but got %s instead !",
               ShapeUtils::shapeAsString(expectedWhFWshape).c_str(), ShapeUtils::shapeAsString(WhFW).c_str());
  REQUIRE_TRUE(WhBW->isSameShape(expectedWhBWshape), 0,
               "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of hidden-to-hidden weights array (for backward "
               "RNN), expected is %s but got %s instead !",
               ShapeUtils::shapeAsString(expectedWhBWshape).c_str(), ShapeUtils::shapeAsString(WhBW).c_str());
  REQUIRE_TRUE(bFW->isSameShape(expectedbFWshape), 0,
               "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of biases array (for forward  RNN), expected is "
               "%s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedbFWshape).c_str(), ShapeUtils::shapeAsString(bFW).c_str());
  REQUIRE_TRUE(bBW->isSameShape(expectedbBWshape), 0,
               "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of biases array (for backward RNN), expected is "
               "%s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedbBWshape).c_str(), ShapeUtils::shapeAsString(bBW).c_str());
  if (h0FW) {
    const std::vector<LongType> expectedh0FWshape = {bS, numUnitsFW};
    REQUIRE_TRUE(h0FW->isSameShape(expectedh0FWshape), 0,
                 "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of initial cell output array (for forward  "
                 "RNN), expected is %s but got %s instead !",
                 ShapeUtils::shapeAsString(expectedh0FWshape).c_str(), ShapeUtils::shapeAsString(h0FW).c_str());
  }
  if (h0BW) {
    const std::vector<LongType> expectedh0BWshape = {bS, numUnitsBW};
    REQUIRE_TRUE(h0BW->isSameShape(expectedh0BWshape), 0,
                 "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of initial cell output array (for backward "
                 "RNN), expected is %s but got %s instead !",
                 ShapeUtils::shapeAsString(expectedh0BWshape).c_str(), ShapeUtils::shapeAsString(h0BW).c_str());
  }
  if (maxTimeStep)
  REQUIRE_TRUE(maxTimeStep->isSameShape({bS}), 0,
               "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of maxTimeStep array, expected is [%i], but "
               "got %s instead !",
               bS, ShapeUtils::shapeAsString(maxTimeStep).c_str());

  // forward steps
  std::vector<sd::LongType> expectedHshape = {time, bS, numUnitsFW + numUnitsBW};
  auto hFW = new NDArray(x->ordering(),expectedHshape, x->dataType(), block.launchContext());
  helpers::rnnTimeLoop(block.launchContext(), x, WxFW, WhFW, bFW, h0FW, maxTimeStep, hFW, hFWFinal);

  auto seqLen = maxTimeStep;
  if (seqLen == nullptr) {
    std::vector<sd::LongType> seqShape = {x->sizeAt(1)};
    seqLen = new NDArray(x->ordering(),seqShape, INT64, block.launchContext());  // [bS]
    *seqLen = x->sizeAt(0);  // set each element of seqLen to be equal to time
  }

  // reverse x
  auto revOut = new NDArray(x, false, block.launchContext());
  helpers::reverseSequence(block.launchContext(), x, seqLen, revOut, 0, 1);

  std::vector<sd::LongType> shape = {time, bS, numUnitsBW};
  // backward steps
  auto hBW = new NDArray(x->ordering(),shape, x->dataType(), block.launchContext());

  helpers::rnnTimeLoop(block.launchContext(), revOut, WxBW, WhBW, bBW, h0BW, maxTimeStep, hBW, hBWFinal);

  // reverse hBW
  auto hBWcopy = new NDArray(*hBW);
  helpers::reverseSequence(block.launchContext(), hBWcopy, seqLen, hBW, 0, 1);

  // concatenate hFW and hBW along last third dimension
  // NDArrayFactory<T>::concat({hFW, hBW}, 2, h);
  helpers::concat(block.launchContext(), {hFW, hBW}, *h, 2);

  delete hBW;
  delete hFW;
  delete hBWcopy;
  delete revOut;

  if (seqLen != maxTimeStep) delete seqLen;

  return Status::OK;
}

DECLARE_TYPES(static_bidirectional_rnn) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(static_bidirectional_rnn) {
  auto xShapeInfo = inputShape->at(0);     // input [time x bS x inSize]
  auto WxFWShapeInfo = inputShape->at(1);  // input-to-hidden  weights for forward  RNN, [inSize  x numUnitsFW]
  auto WhFWShapeInfo = inputShape->at(2);  // hidden-to-hidden weights for forward  RNN, [numUnitsFW x numUnitsFW]
  auto bFWShapeInfo = inputShape->at(3);   // biases for forward  RNN, [2*numUnitsFW]
  auto WxBWShapeInfo = inputShape->at(4);  // input-to-hidden  weights for backward RNN, [inSize  x numUnitsBW]
  auto WhBWShapeInfo = inputShape->at(5);  // hidden-to-hidden weights for backward RNN, [numUnitsBW x numUnitsBW]
  auto bBWShapeInfo = inputShape->at(6);   // biases for backward RNN, [2*numUnitsBW]

  LongType const* h0FWShapeInfo =
      nullptr;  // initial cell output for forward  RNN (at time step = 0) [bS x numUnitsFW]
  LongType const* h0BWShapeInfo =
      nullptr;  // initial cell output for backward RNN (at time step = 0) [bS x numUnitsBW]
  LongType const* maxTimeStepShapeInfo =
      nullptr;  // vector [bS] containing integer values within [0,time), each element of this vector set max time step
  // per each input in batch, this means there are no calculations for time >= maxTimeStep

  switch (block.width()) {
    case 8:
      maxTimeStepShapeInfo = inputShape->at(7);
      break;
    case 9:
      h0FWShapeInfo = inputShape->at(7);
      h0BWShapeInfo = inputShape->at(8);
      break;
    case 10:
      h0FWShapeInfo = inputShape->at(7);
      h0BWShapeInfo = inputShape->at(8);
      maxTimeStepShapeInfo = inputShape->at(9);
      break;
  }

  REQUIRE_TRUE(xShapeInfo[0] == 3, 0,
               "STATIC_BIDIRECTIONAL_RNN custom operation: input array must have rank = 3, but got %i instead !",
               xShapeInfo[0]);
  REQUIRE_TRUE(WxFWShapeInfo[0] == 2, 0,
               "STATIC_BIDIRECTIONAL_RNN custom operation: input-to-hidden weights array (for forward  RNN) must have "
               "rank = 2, but got %i instead !",
               WxFWShapeInfo[0]);
  REQUIRE_TRUE(WxBWShapeInfo[0] == 2, 0,
               "STATIC_BIDIRECTIONAL_RNN custom operation: input-to-hidden weights array (for backward RNN) must have "
               "rank = 2, but got %i instead !",
               WxBWShapeInfo[0]);

  const int inRank = xShapeInfo[0];
  const int time = xShapeInfo[1];
  const int bS = xShapeInfo[2];
  const int numUnitsFW = WxFWShapeInfo[2];
  const int numUnitsBW = WxBWShapeInfo[2];

  const std::vector<LongType> expectedWhFWshape = {numUnitsFW, numUnitsFW};
  const std::vector<LongType> expectedWhBWshape = {numUnitsBW, numUnitsBW};
  const std::vector<LongType> expectedbFWshape = {2 * numUnitsFW};
  const std::vector<LongType> expectedbBWshape = {2 * numUnitsBW};

  REQUIRE_TRUE(ShapeUtils::areShapesEqual(WhFWShapeInfo, expectedWhFWshape), 0,
               "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of hidden-to-hidden weights array (for forward  "
               "RNN), expected is %s but got %s instead !",
               ShapeUtils::shapeAsString(expectedWhFWshape).c_str(), ShapeUtils::shapeAsString(WhFWShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(WhBWShapeInfo, expectedWhBWshape), 0,
               "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of hidden-to-hidden weights array (for backward "
               "RNN), expected is %s but got %s instead !",
               ShapeUtils::shapeAsString(expectedWhBWshape).c_str(), ShapeUtils::shapeAsString(WhBWShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(bFWShapeInfo, expectedbFWshape), 0,
               "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of biases array (for forward  RNN), expected is "
               "%s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedbFWshape).c_str(), ShapeUtils::shapeAsString(bFWShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(bBWShapeInfo, expectedbBWshape), 0,
               "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of biases array (for backward RNN), expected is "
               "%s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedbBWshape).c_str(), ShapeUtils::shapeAsString(bBWShapeInfo).c_str());
  if (h0FWShapeInfo) {
    const std::vector<LongType> expectedh0FWshape = {bS, numUnitsFW};
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(h0FWShapeInfo, expectedh0FWshape), 0,
                 "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of initial cell output array (for forward  "
                 "RNN), expected is %s but got %s instead !",
                 ShapeUtils::shapeAsString(expectedh0FWshape).c_str(),
                 ShapeUtils::shapeAsString(h0FWShapeInfo).c_str());
  }
  if (h0BWShapeInfo) {
    const std::vector<LongType> expectedh0BWshape = {bS, numUnitsBW};
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(h0BWShapeInfo, expectedh0BWshape), 0,
                 "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of initial cell output array (for backward "
                 "RNN), expected is %s but got %s instead !",
                 ShapeUtils::shapeAsString(expectedh0BWshape).c_str(),
                 ShapeUtils::shapeAsString(h0BWShapeInfo).c_str());
  }
  if (maxTimeStepShapeInfo)
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(maxTimeStepShapeInfo, {bS}), 0,
               "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of maxTimeStep array, expected is [%i], but "
               "got %s instead !",
               bS, ShapeUtils::shapeAsString(maxTimeStepShapeInfo).c_str());

  // evaluate output shapeInfos
  LongType *hShapeInfo(nullptr), *hFWFinalPrevShapeInfo(nullptr), *hBWFinalPrevShapeInfo(nullptr);
  ALLOCATE(hShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inRank), sd::LongType);
  ALLOCATE(hFWFinalPrevShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inRank - 1), sd::LongType);
  ALLOCATE(hBWFinalPrevShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inRank - 1), sd::LongType);

  hShapeInfo[0] = inRank;
  hFWFinalPrevShapeInfo[0] = hBWFinalPrevShapeInfo[0] = inRank - 1;
  hShapeInfo[1] = time;
  hShapeInfo[2] = hFWFinalPrevShapeInfo[1] = hBWFinalPrevShapeInfo[1] = bS;
  hShapeInfo[3] = numUnitsFW + numUnitsBW;
  hFWFinalPrevShapeInfo[2] = numUnitsFW;
  hBWFinalPrevShapeInfo[2] = numUnitsBW;

  ShapeUtils::updateStridesAndType(hShapeInfo, xShapeInfo, shape::order(xShapeInfo));
  ShapeUtils::updateStridesAndType(hFWFinalPrevShapeInfo, xShapeInfo, shape::order(xShapeInfo));
  ShapeUtils::updateStridesAndType(hBWFinalPrevShapeInfo, xShapeInfo, shape::order(xShapeInfo));

  return SHAPELIST(CONSTANT(hShapeInfo), CONSTANT(hFWFinalPrevShapeInfo), CONSTANT(hBWFinalPrevShapeInfo));
}

}  // namespace ops
}  // namespace sd
#endif
