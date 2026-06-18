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
#include <ops/declarable/headers/shape.h>
#include <ops/declarable/helpers/shapeOpsHelper.h>

#if NOT_EXCLUDED(OP_shape_of)


namespace sd {
namespace ops {

CUSTOM_OP_IMPL(shape_of, 1, 1, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  const int rank = x->rankOf();

  // Guard: output buffer must be large enough for input rank
  REQUIRE_TRUE(z->lengthOf() >= rank, 0,
      "shape_of: output length (%lld) < input rank (%d)", z->lengthOf(), rank);

  // Delegate to platform-specific helper:
  // CPU: writes to host buffer directly
  // CUDA: uses D2D operations (capturable by CUDA graphs)
  helpers::shapeOfHelper(block.launchContext(), x, z, rank);

  STORE_RESULT(z);

  return Status::OK;
}

DECLARE_SYN(shape, shape_of);

DECLARE_SHAPE_FN(shape_of) {
  auto inShape = inputShape->at(0);

  // LONG by default
  auto dtype = INT64;
  if (block.numI() > 0) dtype = DataTypeUtils::fromInt(INT_ARG(0));

  return SHAPELIST(ConstantShapeHelper::getInstance().vectorShapeInfo(shape::rank(inShape), dtype));
}

DECLARE_TYPES(shape_of) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_INTS})
      ->addTraits(OP_TRAIT_SHAPE_ONLY_OUTPUT | OP_TRAIT_CONSTANT_GENERATION | OP_TRAIT_FULLY_WRITING);
}

}  // namespace ops
}  // namespace sd

#endif

#if NOT_EXCLUDED(OP_set_shape)

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(set_shape, 2, 1, true, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto shape = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);
  REQUIRE_TRUE(shape->isVector() || shape->isScalar(), 0, "Shape must be either a scalar or a vector");
  auto newShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(x->dataType(), x->ordering(),
                                                                         shape->asVectorT<LongType>());
  z->setShapeInfo(newShapeInfo);
  // if x and z aren't the same reference ensure the elements are the same.
  // this op should almost always be used in place and in very specific circumstances.
  if (x != z) {
    z->assign(x, true);
  }
  return Status::OK;
}

DECLARE_SHAPE_FN(set_shape) {
  auto inShape = INPUT_VARIABLE(1);
  return SHAPELIST(inShape->shapeInfo());
}

DECLARE_TYPES(set_shape) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, INT64)
      ->setAllowedOutputTypes({ANY});
}


}  // namespace ops
}  // namespace sd
#endif
