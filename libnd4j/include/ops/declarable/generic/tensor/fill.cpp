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
#if NOT_EXCLUDED(OP_fill)

#include <ops/declarable/headers/parity_ops.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(fill, 1, 1, false, -2, 0) {
  auto shapeArray = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto w = block.width();
  auto i = block.numI();
  auto t = block.numT();

  REQUIRE_TRUE(w > 1 || t > 0 || i > 0, 0,
               "Fill: either additional variable should exist, or scalar value should be present");

  if (output->isEmpty()) {
    // Empty output array - no-op
    return Status::OK;
  }

  if (w > 1) {
    output->assign(INPUT_VARIABLE(1));
  } else {
    if (t > 0) {
      output->assign(T_ARG(0));
    } else if (i > 0) {
      output->assign(INT_ARG(0));
    }
  }

  STORE_RESULT(output);

  return Status::OK;
};

DECLARE_TYPES(fill) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})
      ->setAllowedInputTypes(1, {ALL_INTS, ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_INTS, ALL_FLOATS});
}

DECLARE_SHAPE_FN(fill) {
  auto shapeArray = INPUT_VARIABLE(0);

  const LongType len = shapeArray->lengthOf();
  if (shapeArray->isEmpty()) {
    std::vector<LongType> shape = {0};
    return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(shapeArray->dataType()));
  }
  LongType *newShape = nullptr;
  ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(len), sd::LongType);

  newShape[0] = len;
  bool hasZeros = false;
  LongType totalLen = 1;
  for (int e = 0; e < shapeArray->lengthOf(); e++) {
    newShape[e + 1] = shapeArray->e<LongType>(e);
    if(newShape[e + 1] == 0)
      hasZeros = true;
    totalLen *= newShape[e + 1];
  }
  if(len > 1 && hasZeros) {
    std::vector<LongType> shapeOnly = shapeArray->asVectorT<LongType>();
    return SHAPELIST(ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(shapeArray->dataType(),shapeOnly));
  }
  if (totalLen < 1) {
    std::vector<LongType> shape = {0};
    return SHAPELIST(ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(shapeArray->dataType(), shape));
  }

  DataType dataType;

  if (block.width() > 1) {
    dataType = INPUT_VARIABLE(1)->dataType();
  } else if (block.numT() > 0) {
    dataType = Environment::getInstance().defaultFloatDataType();
  } else if (block.numI() > 0) {
    dataType = INT32;
  } else if (block.numB() > 0) {
    dataType = BOOL;
  } else
    THROW_EXCEPTION("Fill: missing value to fill output array with");

  ShapeUtils::updateStridesAndType(newShape, dataType, 'c');

  return SHAPELIST(CONSTANT(newShape));
};
}  // namespace ops
}  // namespace sd

#endif
