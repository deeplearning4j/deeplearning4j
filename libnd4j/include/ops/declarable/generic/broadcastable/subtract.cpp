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
#if NOT_EXCLUDED(OP_subtract)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/BroadcastHelper.h>

namespace sd {
namespace ops {
BROADCASTABLE_OP_IMPL(subtract, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  BROADCAST_CHECK_EMPTY(x, y, z);

  auto tZ = BroadcastHelper::broadcastApply(BroadcastOpsTuple::Subtract(), x, y, z);
  if (tZ == nullptr)
    return Status::KERNEL_FAILURE;
  else if (tZ != z) {
    OVERWRITE_RESULT(tZ);
  }

  return Status::OK;
}
DECLARE_SYN(Sub, subtract);
DECLARE_SYN(sub, subtract);

DECLARE_TYPES(subtract) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, ANY)
      ->setAllowedOutputTypes(0, INHERIT);
}

CUSTOM_OP_IMPL(subtract_bp, 3, 2, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto epsNext = INPUT_VARIABLE(2);

  auto gradX = OUTPUT_VARIABLE(0);
  auto gradY = OUTPUT_VARIABLE(1);

  if (x->isSameShape(y)) {
    // PWT case case
    epsNext->applyTransform(transform::Neg, gradY);
    gradX->assign(*epsNext);
  } else if (y->isScalar()) {
    // scalar case
    auto tmp = epsNext->reduceNumber(reduce::Sum);
    gradY->assign(-tmp);
    gradX->assign(*epsNext);
  } else {
    // broadcastable
    auto axisX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
    auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

    if (axisX.size() > 0) {
      auto sum = epsNext->reduceAlongDimension(reduce::Sum, &axisX);
      gradX->assign(sum);
    } else
      gradX->assign(*epsNext);

    if (axisY.size() > 0) {
      auto sum = epsNext->reduceAlongDimension(reduce::Sum, &axisY);
      sum.applyTransform(transform::Neg, gradY);
    } else {
      epsNext->applyTransform(transform::Neg, gradY);
    }
  }

  return Status::OK;
}

DECLARE_TYPES(subtract_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(subtract_bp) {
  auto x = inputShape->at(0);
  auto y = inputShape->at(1);
  auto e = inputShape->at(2);

  // eps always has shape of x
  // grad always has shape of y

  LongType *shapeE;
  LongType *shapeG;

  COPY_SHAPE(x, shapeE);
  COPY_SHAPE(y, shapeG);

  auto shapeList = SHAPELIST(CONSTANT(shapeE), CONSTANT(shapeG));

  return shapeList;
}
}  // namespace ops
}  // namespace sd

#endif
