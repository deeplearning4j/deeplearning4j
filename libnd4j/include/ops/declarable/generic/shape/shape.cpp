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
#if NOT_EXCLUDED(OP_shape_of)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(shape_of, 1, 1, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  for (int e = 0; e < x->rankOf(); e++) z->p(e, x->sizeAt(e));

  STORE_RESULT(z);

  return sd::Status::OK;
};
DECLARE_SYN(shape, shape_of);

DECLARE_SHAPE_FN(shape_of) {
  auto inShape = inputShape->at(0);

  // LONG by default
  auto dtype = DataType::INT64;
  if (block.numI() > 0) dtype = DataTypeUtils::fromInt(INT_ARG(0));

  return SHAPELIST(ConstantShapeHelper::getInstance().vectorShapeInfo(shape::rank(inShape), dtype));
};

DECLARE_TYPES(shape_of) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_INTS});
}
#endif

#if NOT_EXCLUDED(OP_set_shape)
CUSTOM_OP_IMPL(set_shape, 2, 1, true, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto shape = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);
  REQUIRE_TRUE(shape->isVector() || shape->isScalar(), 0, "Shape must be either a scalar or a vector");
  auto newShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(x->dataType(), x->ordering(),
                                                                         shape->asVectorT<sd::LongType>());
  z->setShapeInfo(newShapeInfo);
  // if x and z aren't the same reference ensure the elements are the same.
  // this op should almost always be used in place and in very specific circumstances.
  if (x != z) {
    z->assign(x, true);
  }
  return sd::Status::OK;
};

DECLARE_SHAPE_FN(set_shape) {
  auto inShape = INPUT_VARIABLE(1);
  return SHAPELIST(inShape->shapeInfo());
};

DECLARE_TYPES(set_shape) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, sd::DataType::ANY)
      ->setAllowedInputTypes(1, sd::DataType::INT64)
      ->setAllowedOutputTypes({sd::DataType::ANY});
}


#endif

}  // namespace ops
}  // namespace sd