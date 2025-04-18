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
#if NOT_EXCLUDED(OP_divide)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/BroadcastHelper.h>

namespace sd {
namespace ops {
BROADCASTABLE_OP_IMPL(divide, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  BROADCAST_CHECK_EMPTY(x, y, z);

  REQUIRE_TRUE(!y->isB(), 0, "DIVIDE OP: you can't divide by bool array!");
  auto tZ = BroadcastHelper::broadcastApply(BroadcastOpsTuple::Divide(), x, y, z);
  if (tZ == nullptr)
    return Status::KERNEL_FAILURE;
  else if (tZ != z) {
    OVERWRITE_RESULT(tZ);
  }

  return Status::OK;
}
DECLARE_SYN(Div, divide);

DECLARE_TYPES(divide) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, ANY)
      ->setAllowedOutputTypes(0, INHERIT);
}

DECLARE_TYPES(divide_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(divide_bp, 3, 2, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto epsNext = INPUT_VARIABLE(2);

  auto gradX = OUTPUT_VARIABLE(0);
  auto gradY = OUTPUT_VARIABLE(1);


  if (x->isSameShape(y)) {
    // PWT case case

    // X gradient
    NDArray gradXTemp = (*epsNext) / (*y);
    gradX->assign(&gradXTemp);

    // Y gradient
    NDArray gradYTemp = (*epsNext) * (*x) / ((*y) * (*y));
    gradY->assign(&gradYTemp);
    gradY->applyTransform(transform::Neg, gradY);

  } else if (y->isScalar()) {
    // scalar case

    auto tmp = epsNext->reduceNumber(reduce::Sum);
    auto tmpX = x->reduceNumber(reduce::Sum);

    NDArray gradYTemp = tmp * tmpX / ((*y) * (*y));
    gradY->assign(&gradYTemp);
    gradY->applyTransform(transform::Neg, gradY);

    epsNext->applyScalarArr(scalar::Divide, y, gradX);
  } else {
    // broadcast case

    auto preX = *epsNext / *y;

    NDArray negX(*x);
    x->applyTransform(transform::Neg, &negX);
    auto preY = *epsNext * negX / ((*y) * (*y));

    auto axisX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
    auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

    if (axisX.size() > 0) {
      auto sum = preX.reduceAlongDimension(reduce::Sum, &axisX);
      NDArray gradXTemp = sum;
      gradX->assign(&gradXTemp);
    } else {
      NDArray gradXTemp = preX;
      gradX->assign(&gradXTemp);
    }

    if (axisY.size() > 0) {
      auto sum = preY.reduceAlongDimension(reduce::Sum, &axisY);
      NDArray gradYTemp = sum;
      gradY->assign(&gradYTemp);
    } else {
      NDArray gradYTemp = preY;
      gradY->assign(&gradYTemp);
    }
  }

  return Status::OK;
}

DECLARE_SHAPE_FN(divide_bp) {
  auto x = inputShape->at(0);
  auto y = inputShape->at(1);
  auto e = inputShape->at(2);

  // eps always has shape of x
  // grad always has shape of y
  return SHAPELIST(CONSTANT(x), CONSTANT(y));
}
}  // namespace ops
}  // namespace sd

#endif
