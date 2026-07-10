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
#if NOT_EXCLUDED(OP_realdiv)

#include <array/DataTypeUtils.h>
#include <helpers/ConstantShapeHelper.h>
#include <ops/declarable/headers/broadcastable.h>
#include <ops/declarable/generic/helpers/BroadcastHelper.h>

namespace sd {
namespace ops {
BROADCASTABLE_OP_IMPL(realdiv, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  BROADCAST_CHECK_EMPTY(x, y, z);

  // Fast path: same shape - skip BroadcastHelper dispatch overhead
  if (x->isSameShape(y)) {
    x->applyPairwiseTransform(pairwise::Divide, y, z, nullptr);
    return Status::OK;
  }

  // Fast path: scalar divisor - common for normalization
  if (y->isScalar()) {
    x->applyScalarArr(scalar::Divide, y, z);
    return Status::OK;
  }

  auto tZ = BroadcastHelper::broadcastApply(BroadcastOpsTuple::Divide(), x, y, z);
  if (tZ == nullptr) {
    return Status::KERNEL_FAILURE;
  }
  else if (tZ != z) {
    OVERWRITE_RESULT(tZ);
  }

  return Status::OK;
}
DECLARE_SYN(RealDiv, realdiv);

DECLARE_TYPES(realdiv) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, ANY)
      ->setAllowedOutputTypes(0, {FLOAT32, HALF, DOUBLE});
}

DECLARE_TYPES(realdiv_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

static DataType realDivBpGradientType(DataType xType, DataType yType, DataType epsType) {
  auto type = DataTypeUtils::pickPairwiseResultType(xType, yType);
  type = DataTypeUtils::pickPairwiseResultType(type, epsType);
  if (DataTypeUtils::isR(type) && type != DataType::DOUBLE && type != DataType::FLOAT32) return DataType::FLOAT32;
  return type;
}

CUSTOM_OP_IMPL(realdiv_bp, 3, 2, false, 0, 0) {
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
    epsWork->applyPairwiseTransform(pairwise::Divide, yWork, gradX);

    // Y gradient

    // First case
    NDArray negX = -(*xWork);
    NDArray *epsNextMulNegX = (*epsWork) * negX;
    NDArray *ySquared = (*yWork) * (*yWork);
    NDArray *gradYTemp = (*epsNextMulNegX) / (*ySquared);
    gradY->assign(gradYTemp);
    delete epsNextMulNegX;
    delete ySquared;
    delete gradYTemp;

  } else if (y->isScalar()) {
    // scalar case
    NDArray tmp(gradY->dataType(), block.launchContext());
    epsWork->reduceNumber(reduce::Sum, &tmp);
    NDArray tmpX(gradY->dataType(), block.launchContext());
    xWork->reduceNumber(reduce::Sum, &tmpX);

    double tmpVal = tmp.e<double>(0);
    double tmpXVal = tmpX.e<double>(0);
    double yVal = yWork->e<double>(0);
    double gradYVal = -(tmpVal * tmpXVal) / (yVal * yVal);
    gradY->assign(gradYVal);

    epsWork->applyScalarArr(scalar::Divide, yWork, gradX);
  } else {
    // broadcast case

    auto preX = *epsWork / *yWork;

    // Use dup() for a deep copy — NDArray copy constructor creates a VIEW (shares buffer).
    // Writing into a view of x would permanently negate the input variable in-place.
    NDArray *negX = xWork->dup();
    negX->applyTransform(transform::Neg, negX);
    NDArray *epsNextMulNegX = (*epsWork) * (*negX);
    delete negX;
    NDArray *ySquared = (*yWork) * (*yWork);
    NDArray *preY = (*epsNextMulNegX) / (*ySquared);
    delete epsNextMulNegX;
    delete ySquared;

    // Use preX/preY shapes (the broadcast result), NOT epsNext shape —
    // epsNext may be scalar even when x/y are non-scalar.
    auto axisX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), preX->shapeInfo());
    auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), preY->shapeInfo());

    if (axisX.size() > 0) {
      auto sum = preX->reduceAlongDimension(reduce::Sum, &axisX);
      gradX->assign(sum);
      delete sum;
    } else {
      gradX->assign(preX);
    }
    delete preX;

    if (axisY.size() > 0) {
      auto sum = preY->reduceAlongDimension(reduce::Sum, &axisY);
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

DECLARE_SHAPE_FN(realdiv_bp) {
  auto x = inputShape->at(0);
  auto y = inputShape->at(1);
  auto e = inputShape->at(2);

  auto gradType = realDivBpGradientType(ArrayOptions::dataType(x), ArrayOptions::dataType(y), ArrayOptions::dataType(e));
  auto gradXShape = ConstantShapeHelper::getInstance().createShapeInfo(gradType, const_cast<LongType*>(x));
  auto gradYShape = ConstantShapeHelper::getInstance().createShapeInfo(gradType, const_cast<LongType*>(y));
  return SHAPELIST(gradXShape, gradYShape);
}
}  // namespace ops
}  // namespace sd

#endif
