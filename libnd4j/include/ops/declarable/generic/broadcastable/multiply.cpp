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

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {

BROADCASTABLE_OP_IMPL(multiply, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  BROADCAST_CHECK_EMPTY(x, y, z);

   LongType* zShapeInfo = nullptr;
  const bool areShapesBroadcastable =
      ShapeUtils::evalBroadcastShapeInfo(x->shapeInfo(), y->shapeInfo(), true, zShapeInfo, block.getWorkspace());
  REQUIRE_TRUE(areShapesBroadcastable, 0, "MULTIPLY OP: the shapes of x %s and y %s are not suitable for broadcast !",
               ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());

  auto tZ = BroadcastHelper::broadcastApply(BroadcastOpsTuple::Multiply(), x, y, z);
  if (tZ == nullptr)
    return Status::KERNEL_FAILURE;
  else if (tZ != z)
    THROW_EXCEPTION("multiply: result was replaced");

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
    NDArray *dLdxTemp = (*y * *dLdz)->reduceAlongDimension(reduce::Sum, &axesForX);
    dLdx->assign(dLdxTemp);
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
    NDArray *dLdyTemp = (*x * *dLdz)->reduceAlongDimension(reduce::Sum, &axesForY);
    dLdy->assign(dLdyTemp);
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
