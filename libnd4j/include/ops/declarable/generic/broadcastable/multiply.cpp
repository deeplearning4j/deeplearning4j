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
//  @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_multiply)

#include <ops/declarable/headers/broadcastable.h>
#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/helpers/broadcastableFused.h>

namespace sd {
namespace ops {

BROADCASTABLE_OP_IMPL(multiply, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  BROADCAST_CHECK_EMPTY(x, y, z);

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
    const bool xContiguous = x->ordering() == 'c' && shape::strideDescendingCAscendingF(x->shapeInfo());
    const bool yContiguous = y->ordering() == 'c' && shape::strideDescendingCAscendingF(y->shapeInfo());
    const bool zContiguous = z->ordering() == 'c' && shape::strideDescendingCAscendingF(z->shapeInfo());

    if (xContiguous && yContiguous && zContiguous) {
      helpers::fusedMultiplyContiguous(*x, *y, *z);
      cleanupCasts();
      return Status::OK;
    }

    x->applyPairwiseTransform(pairwise::Multiply, y, z, nullptr);
    cleanupCasts();
    return Status::OK;
  }

  const auto xLen = x->lengthOf();
  const auto yLen = y->lengthOf();

  if (yLen == 1) {
    x->applyScalarArr(scalar::Multiply, y, z);
    cleanupCasts();
    return Status::OK;
  }
  if (xLen == 1) {
    y->applyScalarArr(scalar::Multiply, x, z);
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
      // applyBroadcast expects the other array to match TAD shape exactly,
      // so reshape y to strip leading 1s when yRank > 1
      if (yRank > 1) {
        std::vector<sd::LongType> yShape = {yLen};
        auto yReshaped = y->reshape(y->ordering(), yShape);
        x->applyBroadcast(broadcast::Multiply, &dims, yReshaped, z);
        delete yReshaped;
      } else {
        x->applyBroadcast(broadcast::Multiply, &dims, y, z);
      }
      cleanupCasts();
      return Status::OK;
    }
  }
  if (yRank > 1 && xLen == y->sizeAt(-1)) {
    bool compatible = true;
    for (int i = 0; i < xRank - 1; i++) {
      if (x->sizeAt(i) != 1) { compatible = false; break; }
    }
    if (compatible && (xRank == 1 || x->sizeAt(-1) == y->sizeAt(-1))) {
      std::vector<sd::LongType> dims = {yRank - 1};
      if (xRank > 1) {
        std::vector<sd::LongType> xShape = {xLen};
        auto xReshaped = x->reshape(x->ordering(), xShape);
        y->applyBroadcast(broadcast::Multiply, &dims, xReshaped, z);
        delete xReshaped;
      } else {
        y->applyBroadcast(broadcast::Multiply, &dims, x, z);
      }
      cleanupCasts();
      return Status::OK;
    }
  }

  if (xRank == yRank && yRank >= 2) {
    int leadingOnes = 0;
    bool allTrailingMatch = true;
    for (int i = 0; i < yRank; i++) {
      if (y->sizeAt(i) == 1 && x->sizeAt(i) != 1) {
        if (allTrailingMatch && leadingOnes == i) leadingOnes++;
        else { allTrailingMatch = false; break; }
      } else if (y->sizeAt(i) != x->sizeAt(i)) { allTrailingMatch = false; break; }
    }
    if (allTrailingMatch && leadingOnes > 0 && leadingOnes < yRank) {
      std::vector<sd::LongType> dims;
      std::vector<sd::LongType> yShape;
      for (int i = leadingOnes; i < xRank; i++) {
        dims.push_back(i);
        yShape.push_back(y->sizeAt(i));
      }
      // Reshape y to match TAD shape (strip leading 1s)
      auto yReshaped = y->reshape(y->ordering(), yShape);
      x->applyBroadcast(broadcast::Multiply, &dims, yReshaped, z);
      delete yReshaped;
      cleanupCasts();
      return Status::OK;
    }
  }

  if (xRank == yRank && xRank >= 2) {
    int leadingOnes = 0;
    bool allTrailingMatch = true;
    for (int i = 0; i < xRank; i++) {
      if (x->sizeAt(i) == 1 && y->sizeAt(i) != 1) {
        if (allTrailingMatch && leadingOnes == i) leadingOnes++;
        else { allTrailingMatch = false; break; }
      } else if (x->sizeAt(i) != y->sizeAt(i)) { allTrailingMatch = false; break; }
    }
    if (allTrailingMatch && leadingOnes > 0 && leadingOnes < xRank) {
      std::vector<sd::LongType> dims;
      std::vector<sd::LongType> xShape;
      for (int i = leadingOnes; i < yRank; i++) {
        dims.push_back(i);
        xShape.push_back(x->sizeAt(i));
      }
      auto xReshaped = x->reshape(x->ordering(), xShape);
      y->applyBroadcast(broadcast::Multiply, &dims, xReshaped, z);
      delete xReshaped;
      cleanupCasts();
      return Status::OK;
    }
  }

  auto tZ = BroadcastHelper::broadcastApply(BroadcastOpsTuple::Multiply(), x, y, z);
  if (tZ == nullptr) {
    cleanupCasts();
    return Status::KERNEL_FAILURE;
  } else if (tZ != z)
    THROW_EXCEPTION("multiply: result was replaced");

  cleanupCasts();
  return Status::OK;
}
DECLARE_SYN(Mul, multiply);

DECLARE_TYPES(multiply) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, ANY)
      ->setAllowedOutputTypes(0, INHERIT);
}

DECLARE_TYPES(multiply_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

///////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(multiply_bp, 3, 2, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto dLdz = INPUT_VARIABLE(2);

  auto dLdx = OUTPUT_VARIABLE(0);
  auto dLdy = OUTPUT_VARIABLE(1);

   LongType* dLdzShapeInfo = nullptr;
  const bool areShapesBroadcastable =
      ShapeUtils::evalBroadcastShapeInfo(x->shapeInfo(), y->shapeInfo(), true, dLdzShapeInfo, block.getWorkspace());
  REQUIRE_TRUE(areShapesBroadcastable, 0,
               "MULTIPLY_BP OP: the shapes of x %s and y %s are not suitable for broadcast !",
               ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());


  const LongType xLen = x->lengthOf();
  const LongType yLen = y->lengthOf();

  if (x->isScalar() && y->isScalar()) {  // both are scalars
    y->applyPairwiseTransform(pairwise::Multiply, dLdz, dLdx);
    x->applyPairwiseTransform(pairwise::Multiply, dLdz, dLdy);

  }else if (x->isScalar()) {  // x is scalar and y is not
    NDArray *yMulDldz = (*y) * (*dLdz);
    NDArray *dLdxTemp = yMulDldz->reduceNumber(reduce::Sum);
    dLdx->assign(dLdxTemp);
    delete yMulDldz;
    delete dLdxTemp;
    dLdz->applyScalarArr(scalar::Multiply, x, dLdy);
  } else if (y->isScalar()) {  // y is scalar and x is not
    NDArray *xMulDldz = (*x) * (*dLdz);
    NDArray *dLdyTemp = xMulDldz->reduceNumber(reduce::Sum);
    dLdy->assign(dLdyTemp);
    delete xMulDldz;
    delete dLdyTemp;
    dLdz->applyScalarArr(scalar::Multiply, y, dLdx);
  } else if (x->isSameShape(y)) {
    x->applyPairwiseTransform(pairwise::Multiply, dLdz, dLdy);
    y->applyPairwiseTransform(pairwise::Multiply, dLdz, dLdx);
  } else if (x->isSameShape(dLdz)) {
    auto yTiled = NDArray(dLdz, false, block.launchContext());
    y->tile(yTiled);
    std::vector<LongType> axesForY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), dLdz->shapeInfo());

    NDArray *xMulDldz = (*x) * (*dLdz);
    NDArray *dLdyTemp = xMulDldz->reduceAlongDimension(reduce::Sum, &axesForY);
    dLdy->assign(dLdyTemp);
    delete xMulDldz;
    delete dLdyTemp;
    yTiled.applyPairwiseTransform(pairwise::Multiply, dLdz, dLdx);
  } else if (y->isSameShape(dLdz)) {
    auto xTiled = NDArray(dLdz, false, block.launchContext());
    x->tile(xTiled);
    std::vector<LongType> axesForX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), dLdz->shapeInfo());
    // FIXED: Clean up intermediate result from operator*
    NDArray *yMulDldz = (*y) * (*dLdz);
    NDArray *dLdxTemp = yMulDldz->reduceAlongDimension(reduce::Sum, &axesForX);
    dLdx->assign(dLdxTemp);
    delete yMulDldz;
    delete dLdxTemp;
    xTiled.applyPairwiseTransform(pairwise::Multiply, dLdz, dLdy);
  } else {
    auto xTiled = NDArray(dLdz, false, block.launchContext());
    auto yTiled = NDArray(dLdz, false, block.launchContext());
    x->tile(xTiled);
    y->tile(yTiled);
    std::vector<LongType> axesForX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), dLdz->shapeInfo());
    std::vector<LongType> axesForY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), dLdz->shapeInfo());

    // For dLdx
    NDArray *yMulDldz = (*y) * (*dLdz);
    NDArray *dLdxTemp = yMulDldz->reduceAlongDimension(reduce::Sum, &axesForX);
    dLdx->assign(dLdxTemp);
    delete yMulDldz;
    delete dLdxTemp;

    // For dLdy
    // FIXED: Clean up intermediate result from operator*
    NDArray *xMulDldz = (*x) * (*dLdz);
    NDArray *dLdyTemp = xMulDldz->reduceAlongDimension(reduce::Sum, &axesForY);
    dLdy->assign(dLdyTemp);
    delete xMulDldz;
    delete dLdyTemp;
  }

  return Status::OK;
}

DECLARE_SHAPE_FN(multiply_bp) {
  auto xShapeInfo = inputShape->at(0);
  auto yShapeInfo = inputShape->at(1);
  return SHAPELIST(CONSTANT(xShapeInfo), CONSTANT(yShapeInfo));
}

}  // namespace ops
}  // namespace sd

#endif
