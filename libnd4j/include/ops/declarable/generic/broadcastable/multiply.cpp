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

  const sd::LongType* zShapeInfo = nullptr;
  const bool areShapesBroadcastable =
      ShapeUtils::evalBroadcastShapeInfo(x->shapeInfo(), y->shapeInfo(), true, zShapeInfo, block.getWorkspace());
  REQUIRE_TRUE(areShapesBroadcastable, 0, "MULTIPLY OP: the shapes of x %s and y %s are not suitable for broadcast !",
               ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());

  auto tZ = BroadcastHelper::broadcastApply(sd::BroadcastOpsTuple::Multiply(), x, y, z);
  if (tZ == nullptr)
    return sd::Status::KERNEL_FAILURE;
  else if (tZ != z)
    THROW_EXCEPTION("multiply: result was replaced");

  return sd::Status::OK;
}
DECLARE_SYN(Mul, multiply);

DECLARE_TYPES(multiply) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, DataType::ANY)
      ->setAllowedInputTypes(1, DataType::ANY)
      ->setAllowedOutputTypes(0, DataType::INHERIT);
}

DECLARE_TYPES(multiply_bp) {
  getOpDescriptor()->setAllowedInputTypes(DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

///////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(multiply_bp, 3, 2, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto dLdz = INPUT_VARIABLE(2);

  auto dLdx = OUTPUT_VARIABLE(0);
  auto dLdy = OUTPUT_VARIABLE(1);

  const sd::LongType* dLdzShapeInfo = nullptr;
  const bool areShapesBroadcastable =
      ShapeUtils::evalBroadcastShapeInfo(x->shapeInfo(), y->shapeInfo(), true, dLdzShapeInfo, block.getWorkspace());
  REQUIRE_TRUE(areShapesBroadcastable, 0,
               "MULTIPLY_BP OP: the shapes of x %s and y %s are not suitable for broadcast !",
               ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());


  const sd::LongType xLen = x->lengthOf();
  const sd::LongType yLen = y->lengthOf();

  if (x->isScalar() && y->isScalar()) {  // both are scalars
    y->applyPairwiseTransform(pairwise::Multiply, *dLdz, *dLdx);
    x->applyPairwiseTransform(pairwise::Multiply, *dLdz, *dLdy);

  } else if (x->isScalar()) {  // x is scalar and y is not
    dLdx->assign((*y * *dLdz).reduceNumber(reduce::Sum));
    dLdz->applyScalarArr(scalar::Multiply, *x, *dLdy);
  } else if (y->isScalar()) {  // y is scalar and x is not
    dLdy->assign((*x * *dLdz).reduceNumber(reduce::Sum));
    dLdz->applyScalarArr(scalar::Multiply, *y, *dLdx);
  } else if (x->isSameShape(y)) {
    x->applyPairwiseTransform(pairwise::Multiply, *dLdz, *dLdy);
    y->applyPairwiseTransform(pairwise::Multiply, *dLdz, *dLdx);
  } else if (x->isSameShape(dLdz)) {
    auto yTiled = NDArray(dLdz, false, block.launchContext());
    y->tile(yTiled);
    std::vector<sd::LongType> axesForY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), dLdz->shapeInfo());

    dLdy->assign((*x * *dLdz).reduceAlongDimension(reduce::Sum, axesForY));
    yTiled.applyPairwiseTransform(pairwise::Multiply, *dLdz, *dLdx);
  } else if (y->isSameShape(dLdz)) {
    auto xTiled = NDArray(dLdz, false, block.launchContext());
    x->tile(xTiled);
    std::vector<sd::LongType> axesForX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), dLdz->shapeInfo());

    dLdx->assign((*y * *dLdz).reduceAlongDimension(reduce::Sum, axesForX));
    xTiled.applyPairwiseTransform(pairwise::Multiply, *dLdz, *dLdy);
  } else {
    auto xTiled = NDArray(dLdz, false, block.launchContext());
    auto yTiled = NDArray(dLdz, false, block.launchContext());
    x->tile(xTiled);
    y->tile(yTiled);
    std::vector<sd::LongType> axesForX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), dLdz->shapeInfo());
    std::vector<sd::LongType> axesForY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), dLdz->shapeInfo());

    dLdx->assign((*y * *dLdz).reduceAlongDimension(reduce::Sum, axesForX));
    dLdy->assign((*x * *dLdz).reduceAlongDimension(reduce::Sum, axesForY));
  }

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(multiply_bp) {
  auto xShapeInfo = inputShape->at(0);
  auto yShapeInfo = inputShape->at(1);

  sd::LongType* dLdxShapeInfo = nullptr;
  sd::LongType* dLdyShapeInfo = nullptr;

  COPY_SHAPE(xShapeInfo, dLdxShapeInfo);
  COPY_SHAPE(yShapeInfo, dLdyShapeInfo);

  return SHAPELIST(CONSTANT(dLdxShapeInfo), CONSTANT(dLdyShapeInfo));
}

}  // namespace ops
}  // namespace sd

#endif
