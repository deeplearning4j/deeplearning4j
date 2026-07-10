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

#include <array/DataTypeUtils.h>
#include <helpers/ConstantShapeHelper.h>
#include <ops/declarable/headers/broadcastable.h>
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
      ->setAllowedOutputTypes(0, INHERIT)
      ->addTraits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}

DECLARE_TYPES(reversedivide_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

static DataType reverseDivideBpGradientType(DataType xType, DataType yType, DataType epsType) {
  auto type = DataTypeUtils::pickPairwiseResultType(xType, yType);
  type = DataTypeUtils::pickPairwiseResultType(type, epsType);
  if (DataTypeUtils::isR(type) && type != DataType::DOUBLE && type != DataType::FLOAT32) return DataType::FLOAT32;
  return type;
}

CUSTOM_OP_IMPL(reversedivide_bp, 3, 2, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto epsNext = INPUT_VARIABLE(2);

  auto gradX = OUTPUT_VARIABLE(0);
  auto gradY = OUTPUT_VARIABLE(1);

  NDArray *xCast = nullptr, *yCast = nullptr, *epsCast = nullptr;
  NDArray* xWork = x;
  NDArray* yWork = y;
  NDArray* epsWork = epsNext;
  auto gradType = gradX->dataType();
  if (x->dataType() != gradType) {
    xCast = x->cast(gradType);
    xWork = xCast;
  }
  if (y->dataType() != gradType) {
    yCast = y->cast(gradType);
    yWork = yCast;
  }
  if (epsNext->dataType() != gradType) {
    epsCast = epsNext->cast(gradType);
    epsWork = epsCast;
  }
  auto cleanupCasts = [&]() { delete xCast; delete yCast; delete epsCast; };

  if (x->isSameShape(y)) {
    // PWT case case

    // X gradient
    auto* epsY = (*epsWork) * (*yWork);
    auto* xSquared = (*xWork) * (*xWork);
    auto* gradXTemp = (*epsY) / (*xSquared);
    delete epsY;
    delete xSquared;
    gradX->assign(gradXTemp);
    delete gradXTemp;
    gradX->applyTransform(transform::Neg, gradX);

    // Y gradient
    auto* gradYTemp = (*epsWork) / (*xWork);
    gradY->assign(gradYTemp);
    delete gradYTemp;
  } else if (y->isScalar()) {
    // scalar case
    auto* tmp = epsWork->reduceNumber(reduce::Sum);
    auto* tmpX = xWork->reduceNumber(reduce::Sum);
    // For gradY
    auto* gradYTemp = (*tmp) / (*tmpX);
    delete tmp;
    delete tmpX;
    gradY->assign(gradYTemp);
    delete gradYTemp;

    // For gradX
    auto* epsY = (*epsWork) * (*yWork);
    auto* xSquared = (*xWork) * (*xWork);
    auto* gradXTemp = (*epsY) / (*xSquared);
    delete epsY;
    delete xSquared;
    gradX->assign(gradXTemp);
    delete gradXTemp;
    gradX->applyTransform(transform::Neg, gradX);
  } else {
    // broadcast case

    auto* preY = (*epsWork) / (*xWork);

    auto* epsY = (*epsWork) * (*yWork);
    auto* xSquared = (*xWork) * (*xWork);
    auto* preXTemp = (*epsY) / (*xSquared);
    delete epsY;
    delete xSquared;
    preXTemp->applyTransform(transform::Neg, preXTemp);

    // Use preXTemp/preY shapes (the broadcast result), NOT epsNext shape —
    // epsNext may be scalar even when x/y are non-scalar.
    auto axisX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), preXTemp->shapeInfo());
    auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), preY->shapeInfo());

    if (axisX.size() > 0) {
      auto* sum = preXTemp->reduceAlongDimension(reduce::Sum, &axisX);
      gradX->assign(sum);
      delete sum;
    } else {
      gradX->assign(preXTemp);
    }
    delete preXTemp;

    if (axisY.size() > 0) {
      auto* sum = preY->reduceAlongDimension(reduce::Sum, &axisY);
      gradY->assign(sum);
      delete sum;
    } else {
      gradY->assign(preY);
    }
    delete preY;
  }

  cleanupCasts();
  return Status::OK;
}

DECLARE_SHAPE_FN(reversedivide_bp) {
  auto x = inputShape->at(0);
  auto y = inputShape->at(1);
  auto e = inputShape->at(2);
  auto gradType = reverseDivideBpGradientType(ArrayOptions::dataType(x), ArrayOptions::dataType(y), ArrayOptions::dataType(e));
  auto gradXShape = ConstantShapeHelper::getInstance().createShapeInfo(gradType, const_cast<LongType*>(x));
  auto gradYShape = ConstantShapeHelper::getInstance().createShapeInfo(gradType, const_cast<LongType*>(y));
  return SHAPELIST(gradXShape, gradYShape);
}
}  // namespace ops
}  // namespace sd

#endif
