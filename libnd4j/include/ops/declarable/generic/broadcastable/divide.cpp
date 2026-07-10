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

#include <array/DataTypeUtils.h>
#include <helpers/ConstantShapeHelper.h>
#include <ops/declarable/headers/broadcastable.h>
#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/helpers/broadcastableFused.h>

namespace sd {
namespace ops {
BROADCASTABLE_OP_IMPL(divide, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  BROADCAST_CHECK_EMPTY(x, y, z);

  REQUIRE_TRUE(!y->isB(), 0, "DIVIDE OP: you can't divide by bool array!");

  // When input types differ, cast to output type to avoid type-punning in kernels
  NDArray *castX = nullptr, *castY = nullptr;
  auto cleanupCasts = [&]() { delete castX; delete castY; };
  if (x->dataType() != z->dataType()) {
    castX = x->cast(z->dataType());
    x = castX;
  }
  if (y->dataType() != z->dataType()) {
    castY = y->cast(z->dataType());
    y = castY;
  }

  // Fast path: same shape - skip BroadcastHelper dispatch overhead
  if (x->isSameShape(y)) {
    const bool xContiguous = x->ordering() == 'c' && shape::strideDescendingCAscendingF(x->shapeInfo()) && !shape::isViewConst(x->shapeInfo());
    const bool yContiguous = y->ordering() == 'c' && shape::strideDescendingCAscendingF(y->shapeInfo()) && !shape::isViewConst(y->shapeInfo());
    const bool zContiguous = z->ordering() == 'c' && shape::strideDescendingCAscendingF(z->shapeInfo()) && !shape::isViewConst(z->shapeInfo());

    if (xContiguous && yContiguous && zContiguous) {
      helpers::fusedDivideContiguous(*x, *y, *z);
      cleanupCasts();
      return Status::OK;
    }

    x->applyPairwiseTransform(pairwise::Divide, y, z, nullptr);
    cleanupCasts();
    return Status::OK;
  }

  const auto xLen = x->lengthOf();
  const auto yLen = y->lengthOf();

  if (yLen == 1) {
    x->applyScalarArr(scalar::Divide, y, z);
    cleanupCasts();
    return Status::OK;
  }

  const auto xRank = x->rankOf();
  const auto yRank = y->rankOf();

  if (xRank > 1 && yLen == x->sizeAt(-1)) {
    bool compatible = true;
    for (int i = 0; i < yRank - 1; i++) {
      if (y->sizeAt(i) != 1) { compatible = false; break; }
    }
    if (compatible && (yRank == 1 || y->sizeAt(-1) == x->sizeAt(-1))) {
      std::vector<sd::LongType> dims = {xRank - 1};
      if (yRank > 1) {
        std::vector<sd::LongType> yShape = {yLen};
        auto yReshaped = y->reshape(y->ordering(), yShape, false);
        x->applyBroadcast(broadcast::Divide, &dims, yReshaped, z);
        delete yReshaped;
      } else {
        x->applyBroadcast(broadcast::Divide, &dims, y, z);
      }
      cleanupCasts();
      return Status::OK;
    }
  }

  auto tZ = BroadcastHelper::broadcastApply(BroadcastOpsTuple::Divide(), x, y, z);
  if (tZ == nullptr) {
    cleanupCasts();
    return Status::KERNEL_FAILURE;
  } else if (tZ != z) {
    OVERWRITE_RESULT(tZ);
  }

  cleanupCasts();
  return Status::OK;
}
DECLARE_SYN(Div, divide);

DECLARE_TYPES(divide) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, ANY)
      ->setAllowedOutputTypes(0, INHERIT)
      ->addTraits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}

DECLARE_TYPES(divide_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

static DataType divideBpGradientType(DataType xType, DataType yType, DataType epsType) {
  auto type = DataTypeUtils::pickPairwiseResultType(xType, yType);
  type = DataTypeUtils::pickPairwiseResultType(type, epsType);
  if (DataTypeUtils::isR(type) && type != DataType::DOUBLE && type != DataType::FLOAT32) return DataType::FLOAT32;
  return type;
}

CUSTOM_OP_IMPL(divide_bp, 3, 2, false, 0, 0) {
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

  if (x->isSameShape(y) && epsNext->isSameShape(x)) {
    // PWT case: all three inputs have the same shape

    // X gradient: gradX = epsNext / y
    epsWork->applyPairwiseTransform(pairwise::Divide, yWork, gradX);
    // Y gradient: gradY = -(epsNext * x) / (y * y)
    NDArray numerator(epsWork->shapeInfo(), false, block.launchContext());
    epsWork->applyPairwiseTransform(pairwise::Multiply, xWork, &numerator);
    NDArray denominator(yWork->shapeInfo(), false, block.launchContext());
    yWork->applyPairwiseTransform(pairwise::Multiply, yWork, &denominator);
    numerator.applyPairwiseTransform(pairwise::Divide, &denominator, gradY);
    gradY->applyTransform(transform::Neg, gradY);

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
    // broadcast case — epsNext, x, y may have different shapes

    auto preX = *epsWork / *yWork;

    // Use dup() for a deep copy — NDArray copy constructor creates a VIEW (shares buffer).
    // Writing into a view of x would permanently negate the input variable in-place.
    NDArray *negX = xWork->dup();
    negX->applyTransform(transform::Neg, negX);
    NDArray *negXMulEps = (*epsWork) * (*negX);
    NDArray *ySquared = (*yWork) * (*yWork);
    auto preY = (*negXMulEps) / (*ySquared);
    delete negX;
    delete negXMulEps;
    delete ySquared;

    // Reduce along axes where x/y were broadcast to the pre-gradient shape.
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

DECLARE_SHAPE_FN(divide_bp) {
  auto x = inputShape->at(0);
  auto y = inputShape->at(1);
  auto e = inputShape->at(2);

  auto gradType = divideBpGradientType(ArrayOptions::dataType(x), ArrayOptions::dataType(y), ArrayOptions::dataType(e));
  auto gradXShape = ConstantShapeHelper::getInstance().createShapeInfo(gradType, const_cast<LongType*>(x));
  auto gradYShape = ConstantShapeHelper::getInstance().createShapeInfo(gradType, const_cast<LongType*>(y));
  return SHAPELIST(gradXShape, gradYShape);
}
}  // namespace ops
}  // namespace sd

#endif
