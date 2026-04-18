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

    // X gradient: gradX = epsNext / y
    epsNext->applyPairwiseTransform(pairwise::Divide, y, gradX);
    // Y gradient: gradY = -(epsNext * x) / (y * y)
    NDArray numerator(epsNext->shapeInfo(), false, block.launchContext());
    epsNext->applyPairwiseTransform(pairwise::Multiply, x, &numerator);
    NDArray denominator(y->shapeInfo(), false, block.launchContext());
    y->applyPairwiseTransform(pairwise::Multiply, y, &denominator);
    numerator.applyPairwiseTransform(pairwise::Divide, &denominator, gradY);
    gradY->applyTransform(transform::Neg, gradY);

  } else if (y->isScalar()) {
    // scalar case

    NDArray tmp(gradY->dataType(), block.launchContext());
    epsNext->reduceNumber(reduce::Sum, &tmp);
    NDArray tmpX(gradY->dataType(), block.launchContext());
    x->reduceNumber(reduce::Sum, &tmpX);

    double tmpVal = tmp.e<double>(0);
    double tmpXVal = tmpX.e<double>(0);
    double yVal = y->e<double>(0);
    double gradYVal = -(tmpVal * tmpXVal) / (yVal * yVal);
    gradY->assign(gradYVal);

    epsNext->applyScalarArr(scalar::Divide, y, gradX);
  } else {
    // broadcast case

    auto preX = *epsNext / *y;

    NDArray negX(*x);
    x->applyTransform(transform::Neg, &negX);
    NDArray *negXMulEps = (*epsNext) * negX;
    NDArray *ySquared = (*y) * (*y);
    auto preY = (*negXMulEps) / (*ySquared);
    delete negXMulEps;
    delete ySquared;
    auto axisX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
    auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

    if (axisX.size() > 0) {
      auto sum = preX->reduceAlongDimension(reduce::Sum, &axisX);
      gradX->assign(sum);
      delete sum;
    } else {
      // FIXED: preX is stack-allocated from operator/, don't delete
      gradX->assign(preX);
    }

    if (axisY.size() > 0) {
      auto sum = preY->reduceAlongDimension(reduce::Sum, &axisY);
      gradY->assign(sum);
      delete sum;
    } else {
      // FIXED: preY is stack-allocated from operator/, don't delete
      gradY->assign(preY);
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
