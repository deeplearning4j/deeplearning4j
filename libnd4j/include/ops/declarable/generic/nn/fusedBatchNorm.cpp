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
// Created by raver119 on 29/10/17.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_fused_batch_norm)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {

DECLARE_TYPES(fused_batch_norm) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(fused_batch_norm, 3, 3, false, 0, 2) {
  auto x = INPUT_VARIABLE(0);       // [bS,iH,iW,iD] (NHWC) or [bS,iD,iH,iW] (NCHW)
  auto scale = INPUT_VARIABLE(1);   // [iD]
  auto offset = INPUT_VARIABLE(2);  // [iD]

  auto y = OUTPUT_VARIABLE(0);          // [bS,iH,iW,iD] (NHWC) or [bS,iD,iH,iW] (NCHW)
  auto batchMean = OUTPUT_VARIABLE(1);  // [iD]
  auto batchVar = OUTPUT_VARIABLE(2);   // [iD]

  const bool dataFormat = (bool)INT_ARG(0);  // 0->NHWC, 1->NCHW
  const bool isTraining = (bool)INT_ARG(1);
  sd_debug("CUSTOM_OP fused_batch_norm: data format, is NCHW: %d, isTraining: %d\n", dataFormat, isTraining);

  REQUIRE_TRUE(x->rankOf() == 4, 0,
               "CUSTOM_OP fused_batch_norm: the rank of input x array must be equal to 4, but got %i instead !",
               x->rankOf());

  int  iD;         // input height, input width, input depth(number of channels)
  if (dataFormat) {
    iD = x->sizeAt(1);
  } else {
    iD = x->sizeAt(3);
  }

  auto xCast = x->cast(sd::DataType::FLOAT32);
  if (dataFormat) {
    std::vector<LongType> permute = {0,2,3,1};
    xCast = xCast.permute(permute, false, false);
  }
  REQUIRE_TRUE(scale->rankOf() == 1 && scale->sizeAt(0) == iD, 0,
               "CUSTOM_OP fused_batch_norm: wrong shape of input scale array, expected is [%i], but got %s instead", iD,
               ShapeUtils::shapeAsString(scale).c_str());
  REQUIRE_TRUE(offset->rankOf() == 1 && offset->sizeAt(0) == iD, 0,
               "CUSTOM_OP fused_batch_norm: wrong shape of input offset array, expected is [%i], but got %s instead",
               iD, ShapeUtils::shapeAsString(offset).c_str());

  NDArray *mean(nullptr), *variance(nullptr);
  if (!isTraining) {
    mean = INPUT_VARIABLE(3);
    variance = INPUT_VARIABLE(4);
    REQUIRE_TRUE(mean->rankOf() == 1 && mean->sizeAt(0) == iD, 0,
                 "CUSTOM_OP fused_batch_norm: wrong shape of input mean array, expected is [%i], but got %s instead",
                 iD, ShapeUtils::shapeAsString(mean).c_str());
    REQUIRE_TRUE(
        variance->rankOf() == 1 && variance->sizeAt(0) == iD, 0,
        "CUSTOM_OP fused_batch_norm: wrong shape of input variance array, expected is [%i], but got %s instead", iD,
        ShapeUtils::shapeAsString(variance).c_str());
  } else {
    // REQUIRE_TRUE(block.width() == 3, 0, "CUSTOM_OP fused_batch_norm: when isTraining=true then number of input arrays
    // must be equal to 3, but got %i instead !", block.width());
    std::vector<sd::LongType> shape = {iD};
    mean = NDArrayFactory::create_(scale->ordering(), shape, scale->dataType(), block.launchContext());
    variance = NDArrayFactory::create_(scale->ordering(), shape, scale->dataType(), block.launchContext());
  }

  float epsilon;
  if (block.getTArguments()->size() > 0) {
    epsilon = (float)(T_ARG(0) > 1.001e-5 ? T_ARG(0) : 1.001e-5);
  } else {
    epsilon = 0.001f;
  }

  const int restSize = x->lengthOf() / iD;

  auto xAffected = NDArrayFactory::create(x->ordering(), {restSize, iD}, mean->dataType(), block.launchContext());
  xAffected.assign(&xCast);

  const int restSizeMinusOne = (restSize > 1) ? (restSize - 1) : 1;
  const float restSizeInv = 1.0f / restSize;
  const float restSizeAdjust = (float)restSize / restSizeMinusOne;

  if (isTraining) {
    std::vector<sd::LongType > dim = {0};
    auto sum = xAffected.reduceAlongDimension(reduce::Sum, &dim);
    sum *= restSizeInv;
    mean->assign(&sum);
    *batchMean = *mean;
  } else
    *batchMean = 0.;

  auto xCentered = xAffected - *mean;
  xAffected -= *mean;

  if (isTraining) {
    int power = 2;
    xAffected.applyScalar(scalar::Pow, power, &xAffected);
    std::vector<sd::LongType > dim = {0};

    auto sum = xAffected.reduceAlongDimension(reduce::Sum, &dim);
    sum *= restSizeInv;
    variance->assign(&sum);
    auto varOutput = (*variance) * restSizeAdjust;
    batchVar->assign(&varOutput);
  } else
    *batchVar = 0.;

  auto scaledVariance = ((*variance + epsilon).transform(transform::RSqrt) * (*scale)).cast(xAffected.dataType());
  auto xScaled1 = xCentered * scaledVariance;
  auto xShifted1 = xScaled1 + *offset;
  if (dataFormat) {
    // need to reshape from matrix to 4d then permute the ordering due to NWHC  ordering
    auto newShape = xCast.getShapeAsVector();
    auto reshaped = xShifted1.reshape(xCast.ordering(), newShape,false);
    reshaped.permutei({0, 3, 1, 2}, 0, false);
    y->assign(&reshaped);

  } else  // NWHC case
    y->assign(&xShifted1);

  if (isTraining) {
    delete mean;
    delete variance;
  }

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(fused_batch_norm) {
  auto xShapeInfo = inputShape->at(0);
  auto scaleShapeInfo = inputShape->at(1);

  const bool dataFormat = (bool)INT_ARG(0);  // 0->NHWC, 1->NCHW
  const int iD = dataFormat ? xShapeInfo[2] : xShapeInfo[4];

  REQUIRE_TRUE(scaleShapeInfo[0] == 1 && scaleShapeInfo[1] == iD, 0,
               "CUSTOM_OP fused_batch_norm: wrong shape of input scale array, expected is [%i], but got %s instead", iD,
               ShapeUtils::shapeAsString(scaleShapeInfo).c_str());
  return SHAPELIST(CONSTANT(xShapeInfo), CONSTANT(scaleShapeInfo), CONSTANT(scaleShapeInfo));
}

}  // namespace ops
}  // namespace sd

#endif
