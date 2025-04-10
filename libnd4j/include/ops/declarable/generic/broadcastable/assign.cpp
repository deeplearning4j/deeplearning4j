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
// Created by raver119 on 24.11.17.
//

#include <system/op_boilerplate.h>
#include <helpers/StringUtils.h>
#if NOT_EXCLUDED(OP_assign)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/BroadcastHelper.h>

namespace sd {
namespace ops {
BROADCASTABLE_OP_IMPL(assign, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = block.width() < 2 ? x: INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  // Check if any array is of string type
  if (x->isS() || y->isS() || z->isS()) {
    // Handle string broadcast at high level
    StringUtils::broadcastStringAssign(x,z);
    return Status::OK;
  }

  NDArray *castedX = x->dataType() == z->dataType() ? x : new NDArray(x->cast(z->dataType()));
  NDArray *castedY = y->dataType() == z->dataType() ? y : new NDArray(y->cast(z->dataType()));

  ArrayOptions::validateSingleDataType(ArrayOptions::dataType(castedX->shapeInfo()));
  ArrayOptions::validateSingleDataType(ArrayOptions::extra(castedY->shapeInfo()));
  ArrayOptions::validateSingleDataType(ArrayOptions::extra(z->shapeInfo()));

  auto tZ = BroadcastHelper::broadcastApply(BroadcastOpsTuple::Assign(), castedX, castedY, z);

  if (tZ != z) {
    OVERWRITE_RESULT(tZ);
  }


  return Status::OK;
}
DECLARE_SYN(set, assign);
DECLARE_SYN(copy, assign);

DECLARE_TYPES(assign) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS,ALL_FLOATS,ALL_STRINGS,BOOL})
      ->setAllowedInputTypes(1, {ALL_INTS,ALL_FLOATS,ALL_STRINGS,BOOL})
      ->setAllowedOutputTypes(0, {ALL_INTS,ALL_FLOATS,ALL_STRINGS,BOOL});
}

DECLARE_TYPES(assign_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_INTS,ALL_FLOATS,ALL_STRINGS});
}

CUSTOM_OP_IMPL(assign_bp, 3, 2, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = block.width() < 2 ? new NDArray(x->dup(x->ordering(), false)) : INPUT_VARIABLE(1);
  auto epsNext = INPUT_VARIABLE(2);

  auto gradX = OUTPUT_VARIABLE(0);
  auto gradY = OUTPUT_VARIABLE(1);

  float zero = 0.0f;
  gradX->assign(zero);

  if (x->isSameShape(y)) {
    gradY->assign(epsNext);
  } else if (y->isScalar()) {
    auto sum = epsNext->reduceNumber(reduce::Sum);
    gradY->assign(&sum);
  } else {
    // broadcastable
    auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

    if (axisY.size() > 0) {
      auto sum = epsNext->reduceAlongDimension(reduce::Sum, &axisY);
      gradY->assign(&sum);
    } else
      gradY->assign(epsNext);
  }

  return Status::OK;
}

DECLARE_SHAPE_FN(assign_bp) {
  auto x = inputShape->at(0);
  auto y = inputShape->at(1);
  auto e = inputShape->at(2);

  // eps always has shape of x
  // grad always has shape of y
  auto shapeList = SHAPELIST(CONSTANT(x), CONSTANT(y));

  return shapeList;
}
}  // namespace ops
}  // namespace sd

#endif
