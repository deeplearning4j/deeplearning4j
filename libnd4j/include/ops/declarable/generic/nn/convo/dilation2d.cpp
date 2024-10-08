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
//  @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_dilation2d)

#include <ops/declarable/headers/convo.h>
#include <ops/declarable/helpers/dilation2d.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(dilation2d, 2, 1, false, 0, 1) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);

  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->rankOf() == 4, 0, "Dilation2D: input should be 4D");
  REQUIRE_TRUE(weights->rankOf() == 3, 0, "Dilation2D: weights should be 3D");

  const LongType bS = input->sizeAt(0);
  const LongType iC = input->sizeAt(3);
  const bool isSameShape = INT_ARG(0) == 1;

  REQUIRE_TRUE(input->sizeAt(3) == weights->sizeAt(2), 0,
               "Dilation2D: number of input channels doesn't match number of channels in weights: %i vs %i",
               input->sizeAt(3), weights->sizeAt(2));

  std::vector<sd::LongType> strides(4);
  std::vector<sd::LongType> rates(4);

  if (block.width() > 2) {
    REQUIRE_TRUE(block.width() >= 4, 0, "Dilation2D: number of input arrays should be 4 at least");

    auto r = INPUT_VARIABLE(2);
    auto s = INPUT_VARIABLE(3);

    strides = s->template asVectorT<sd::LongType>();
    rates = r->template asVectorT<sd::LongType>();
  } else {
    REQUIRE_TRUE(block.numI() >= 9, 0, "Dilation2D: number of Int arguments should be 9 at least");

    int e = 1;
    for (int cnt = 0; cnt < 4; cnt++) rates[cnt] = INT_ARG(e++);

    for (int cnt = 0; cnt < 4; cnt++) strides[cnt] = INT_ARG(e++);
  }

  sd::LongType sH = 0, sW = 0;
  sd::LongType dH = 0, dW = 0;
  sd::LongType pH = 0, pW = 0;
  sd::LongType oH = 0, oW = 0;

  helpers::dilation_hw(block.launchContext(), input->shapeInfo(), weights->shapeInfo(), strides, rates, isSameShape,
                       &sH, &sW, &pH, &pW, &dH, &dW, &oH, &oW);

  REQUIRE_TRUE(oH > 0 && oW > 0, 0, "Dilation2D: outY and outX should have positive values, but got [%i, %i] instead",
               oH, oW);

  helpers::dilation2d(block.launchContext(), input, weights, output, sH, sW, pH, pW, dH, dW);

  return sd::Status::OK;
}

DECLARE_TYPES(dilation2d) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(dilation2d) {
  auto input = inputShape->at(0);
  auto weights = inputShape->at(1);

  const int bS = shape::sizeAt(input, static_cast<sd::LongType>(0));
  const int iC = shape::sizeAt(input, static_cast<sd::LongType>(3));
  const bool isSameShape = INT_ARG(0) == 1;

  std::vector<sd::LongType> strides(4);
  std::vector<sd::LongType> rates(4);

  if (block.width() > 2) {
    auto r = INPUT_VARIABLE(2);
    auto s = INPUT_VARIABLE(3);

    strides = s->template asVectorT<sd::LongType>();
    rates = r->template asVectorT<sd::LongType>();
  } else {
    if (block.numI() < 9) {
      auto newShape = ConstantShapeHelper::getInstance().scalarShapeInfo(block.dataType());
      return SHAPELIST(newShape);
    }

    int e = 1;
    for (int cnt = 0; cnt < 4; cnt++) rates[cnt] = INT_ARG(e++);

    for (int cnt = 0; cnt < 4; cnt++) strides[cnt] = INT_ARG(e++);
  }

  sd::LongType sH = 0, sW = 0;
  sd::LongType dH = 0, dW = 0;
  sd::LongType pH = 0, pW = 0;
  sd::LongType oH = 0, oW = 0;

  helpers::dilation_hw(block.launchContext(), input, weights, strides, rates, isSameShape, &sH, &sW, &pH, &pW, &dH, &dW,
                       &oH, &oW);

  std::array<sd::LongType, 4> shape = {{bS, oH, oW, iC}};
  auto newShape =
      ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(weights), 'c', 4, shape.data(),0);
  return SHAPELIST(newShape);
}
}  // namespace ops
}  // namespace sd

#endif
