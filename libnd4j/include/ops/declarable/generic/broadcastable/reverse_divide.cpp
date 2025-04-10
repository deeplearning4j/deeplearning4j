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
#if NOT_EXCLUDED(OP_reversedivide)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/BroadcastHelper.h>

namespace sd {
namespace ops {
BROADCASTABLE_OP_IMPL(reversedivide, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  BROADCAST_CHECK_EMPTY(x, y, z);

  REQUIRE_TRUE(!x->isB(), 0, "REVERSEDIVIDE OP: you can't divide by bool array!");
  x->applyTrueBroadcast(BROADCAST(ReverseDivide), y, z, true);

  return Status::OK;
}
DECLARE_SYN(RDiv, reversedivide);

DECLARE_TYPES(reversedivide) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, ANY)
      ->setAllowedOutputTypes(0, INHERIT);
}

DECLARE_TYPES(reversedivide_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(reversedivide_bp, 3, 2, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto epsNext = INPUT_VARIABLE(2);

  auto gradX = OUTPUT_VARIABLE(0);
  auto gradY = OUTPUT_VARIABLE(1);

  if (x->isSameShape(y)) {
    // PWT case case

    // X gradient
    NDArray gradXTemp = (*epsNext) * (*y) / ((*x) * (*x));
    gradX->assign(&gradXTemp);
    gradX->applyTransform(transform::Neg, gradX);

    // Y gradient
    NDArray gradYTemp = (*epsNext) / (*x);
    gradY->assign(&gradYTemp);
  } else if (y->isScalar()) {
    // scalar case
    auto tmp = epsNext->reduceNumber(reduce::Sum);
    auto tmpX = x->reduceNumber(reduce::Sum);
    // For gradY
    NDArray gradYTemp = tmp / tmpX;
    gradY->assign(&gradYTemp);

    // For gradX
    NDArray gradXTemp = (*epsNext) * (*y) / ((*x) * (*x));
    gradX->assign(&gradXTemp);
    gradX->applyTransform(transform::Neg, gradX);
  } else {
    // broadcast case

    auto preY = (*epsNext) / (*x);

    auto preX = *epsNext * (*y) / ((*x) * (*x));
    preX.applyTransform(transform::Neg, &preX);

    auto axisX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
    auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

    if (axisX.size() > 0) {
      auto sum = preX.reduceAlongDimension(reduce::Sum, &axisX);
      gradX->assign(&sum);
    } else
      gradX->assign(&preX);

    if (axisY.size() > 0) {
      auto sum = preY.reduceAlongDimension(reduce::Sum, &axisY);
      gradY->assign(&sum);
    } else
      gradY->assign(&preY);
  }

  return Status::OK;
}

DECLARE_SHAPE_FN(reversedivide_bp) {
  auto x = inputShape->at(0);
  auto y = inputShape->at(1);
  auto e = inputShape->at(2);
  return SHAPELIST(CONSTANT(x), CONSTANT(y));
}
}  // namespace ops
}  // namespace sd

#endif
