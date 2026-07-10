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
#if NOT_EXCLUDED(OP_add)

#include <array/DataTypeUtils.h>
#include <helpers/ConstantShapeHelper.h>
#include <ops/declarable/headers/broadcastable.h>
#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/helpers/broadcastableFused.h>

namespace sd {
namespace ops {
BROADCASTABLE_OP_IMPL(add, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);
  BROADCAST_CHECK_EMPTY(x, y, z);

  // When input types differ, cast the mismatched input to the output type.
  // All fast-path kernels (fused, pairwiseTransform, broadcast, scalar) use
  // single-type templates and will misread memory if types don't match.
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
  // This is the most common case in BERT (residual connections, etc.)
  if (x->isSameShape(y)) {
    // Ultra-fast path for contiguous same-shape same-type arrays
    const bool xContiguous = x->ordering() == 'c' && shape::strideDescendingCAscendingF(x->shapeInfo()) && !shape::isViewConst(x->shapeInfo());
    const bool yContiguous = y->ordering() == 'c' && shape::strideDescendingCAscendingF(y->shapeInfo()) && !shape::isViewConst(y->shapeInfo());
    const bool zContiguous = z->ordering() == 'c' && shape::strideDescendingCAscendingF(z->shapeInfo()) && !shape::isViewConst(z->shapeInfo());

    if (xContiguous && yContiguous && zContiguous) {
      helpers::fusedAddContiguous(*x, *y, *z);
      cleanupCasts();
      return Status::OK;
    }

    x->applyPairwiseTransform(pairwise::Add, y, z, nullptr);
    cleanupCasts();
    return Status::OK;
  }

  // Fast path: scalar or effectively-scalar (length 1) broadcast
  const auto xLen = x->lengthOf();
  const auto yLen = y->lengthOf();

  if (yLen == 1) {
    x->applyScalarArr(scalar::Add, y, z);
    cleanupCasts();
    return Status::OK;
  }
  if (xLen == 1) {
    y->applyScalarArr(scalar::Add, x, z);
    cleanupCasts();
    return Status::OK;
  }

  // Fast path: 1D-like array broadcast along last dimension (very common in BERT)
  // Handles both rank-1 [hidden] and effectively-1D like [1, hidden] or [1, 1, hidden]
  // Pattern: [batch, seq, hidden] + [hidden] or [1, hidden] -> broadcast along last dim
  const auto xRank = x->rankOf();
  const auto yRank = y->rankOf();

  // Check if y broadcasts along x's last dimension
  if (xRank > 1 && yLen == x->sizeAt(-1)) {
    // Verify y's shape is compatible (all dims except last should be 1 or match)
    bool compatible = true;
    for (int i = 0; i < yRank - 1; i++) {
      if (y->sizeAt(i) != 1) {
        compatible = false;
        break;
      }
    }
    if (compatible && (yRank == 1 || y->sizeAt(-1) == x->sizeAt(-1))) {
      // Ultra-fast fused kernel for contiguous bias add
      const bool xContiguous = x->ordering() == 'c' && shape::strideDescendingCAscendingF(x->shapeInfo()) && !shape::isViewConst(x->shapeInfo());
      const bool yContiguous = shape::strideDescendingCAscendingF(y->shapeInfo()) && !shape::isViewConst(y->shapeInfo());
      const bool zContiguous = z->ordering() == 'c' && shape::strideDescendingCAscendingF(z->shapeInfo()) && !shape::isViewConst(z->shapeInfo());

      if (xContiguous && yContiguous && zContiguous) {
        helpers::fusedAdd1DLast(*x, *y, *z);
        cleanupCasts();
        return Status::OK;
      }

      std::vector<sd::LongType> dims = {xRank - 1};
      if (yRank > 1) {
        std::vector<sd::LongType> yShape = {yLen};
        auto yReshaped = y->reshape(y->ordering(), yShape, false);
        x->applyBroadcast(broadcast::Add, &dims, yReshaped, z);
        delete yReshaped;
      } else {
        x->applyBroadcast(broadcast::Add, &dims, y, z);
      }
      cleanupCasts();
      return Status::OK;
    }
  }

  // Reverse case: x broadcasts along y's last dimension
  if (yRank > 1 && xLen == y->sizeAt(-1)) {
    bool compatible = true;
    for (int i = 0; i < xRank - 1; i++) {
      if (x->sizeAt(i) != 1) {
        compatible = false;
        break;
      }
    }
    if (compatible && (xRank == 1 || x->sizeAt(-1) == y->sizeAt(-1))) {
      std::vector<sd::LongType> dims = {yRank - 1};
      if (xRank > 1) {
        std::vector<sd::LongType> xShape = {xLen};
        auto xReshaped = x->reshape(x->ordering(), xShape, false);
        y->applyBroadcast(broadcast::Add, &dims, xReshaped, z);
        delete xReshaped;
      } else {
        y->applyBroadcast(broadcast::Add, &dims, x, z);
      }
      cleanupCasts();
      return Status::OK;
    }
  }

  // Fast path: y has leading 1s, trailing dims match x
  // Pattern: [batch, seq, hidden] + [1, seq, hidden] -> broadcast along dim 0
  // Or: [batch, seq, hidden] + [1, 1, hidden] -> broadcast along dims 0,1
  if (xRank == yRank && yRank >= 2) {
    int leadingOnes = 0;
    bool allTrailingMatch = true;
    for (int i = 0; i < yRank; i++) {
      if (y->sizeAt(i) == 1 && x->sizeAt(i) != 1) {
        if (allTrailingMatch && leadingOnes == i) {
          leadingOnes++;
        } else {
          allTrailingMatch = false;
          break;
        }
      } else if (y->sizeAt(i) != x->sizeAt(i)) {
        allTrailingMatch = false;
        break;
      }
    }
    if (allTrailingMatch && leadingOnes > 0 && leadingOnes < yRank) {
      std::vector<sd::LongType> dims;
      std::vector<sd::LongType> yShape;
      for (int i = leadingOnes; i < xRank; i++) {
        dims.push_back(i);
        yShape.push_back(y->sizeAt(i));
      }
      auto yReshaped = y->reshape(y->ordering(), yShape, false);
      x->applyBroadcast(broadcast::Add, &dims, yReshaped, z);
      delete yReshaped;
      cleanupCasts();
      return Status::OK;
    }
  }

  // Fast path: x has leading 1s, trailing dims match y (reverse of above)
  if (xRank == yRank && xRank >= 2) {
    int leadingOnes = 0;
    bool allTrailingMatch = true;
    for (int i = 0; i < xRank; i++) {
      if (x->sizeAt(i) == 1 && y->sizeAt(i) != 1) {
        if (allTrailingMatch && leadingOnes == i) {
          leadingOnes++;
        } else {
          allTrailingMatch = false;
          break;
        }
      } else if (x->sizeAt(i) != y->sizeAt(i)) {
        allTrailingMatch = false;
        break;
      }
    }
    if (allTrailingMatch && leadingOnes > 0 && leadingOnes < xRank) {
      std::vector<sd::LongType> dims;
      std::vector<sd::LongType> xShape;
      for (int i = leadingOnes; i < yRank; i++) {
        dims.push_back(i);
        xShape.push_back(x->sizeAt(i));
      }
      auto xReshaped = x->reshape(x->ordering(), xShape, false);
      y->applyBroadcast(broadcast::Add, &dims, xReshaped, z);
      delete xReshaped;
      cleanupCasts();
      return Status::OK;
    }
  }

  // Standard path with broadcasting support
  auto tZ = BroadcastHelper::broadcastApply(BroadcastOpsTuple::Add(), x, y, z);
  if (tZ == nullptr) {
    cleanupCasts();
    return Status::KERNEL_FAILURE;
  } else if (tZ != z && !tZ->isEmpty()) {
    OVERWRITE_RESULT(tZ);
  }
  cleanupCasts();
  return Status::OK;
}

DECLARE_TYPES(add) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, ANY)
      ->setAllowedOutputTypes(ANY);
}

DECLARE_TYPES(add_bp) { getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS}); }

static DataType addBpGradientType(DataType xType, DataType yType, DataType epsType) {
  auto type = DataTypeUtils::pickPairwiseResultType(xType, yType);
  type = DataTypeUtils::pickPairwiseResultType(type, epsType);
  if (DataTypeUtils::isR(type) && type != DataType::DOUBLE && type != DataType::FLOAT32) return DataType::FLOAT32;
  return type;
}

CUSTOM_OP_IMPL(add_bp, 3, 2, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto epsNext = INPUT_VARIABLE(2);
  auto gradX = OUTPUT_VARIABLE(0);
  auto gradY = OUTPUT_VARIABLE(1);

  if (x->isSameShape(y)) {
    // PWT case case
    gradY->assign(epsNext);
    gradX->assign(epsNext);
  } else if (y->isScalar()) {
    // scalar case
    auto tmp = epsNext->reduceNumber(reduce::Sum);
    gradY->assign(tmp);
    gradX->assign(epsNext);
    delete tmp;
  } else {
    // broadcast case
    auto axisX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
    auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

    if (axisX.size() > 0) {
      auto sum = epsNext->reduceAlongDimension(reduce::Sum, &axisX);
      gradX->assign(sum);
      delete sum;
    } else
      gradX->assign(epsNext);

    if (axisY.size() > 0) {
      auto sum = epsNext->reduceAlongDimension(reduce::Sum, &axisY);
      gradY->assign(sum);
      delete sum;
    } else
      gradY->assign(epsNext);
  }



  return Status::OK;
}

DECLARE_SHAPE_FN(add_bp) {
  auto x = inputShape->at(0);
  auto y = inputShape->at(1);
  auto e = inputShape->at(2);

  auto gradType = addBpGradientType(ArrayOptions::dataType(x), ArrayOptions::dataType(y), ArrayOptions::dataType(e));
  auto gradXShape = ConstantShapeHelper::getInstance().createShapeInfo(gradType, const_cast<LongType*>(x));
  auto gradYShape = ConstantShapeHelper::getInstance().createShapeInfo(gradType, const_cast<LongType*>(y));
  return SHAPELIST(gradXShape, gradYShape);
}
}  // namespace ops
}  // namespace sd

#endif
