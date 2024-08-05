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
#if NOT_EXCLUDED(OP_bincount)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/weights.h>

namespace sd {
namespace ops {
DECLARE_TYPES(bincount) {
  getOpDescriptor()
      ->setAllowedInputTypes({ALL_INTS})
      ->setAllowedInputTypes(1, ANY)
      ->setAllowedOutputTypes({ALL_INTS, ALL_FLOATS});
}

CUSTOM_OP_IMPL(bincount, 1, 1, false, 0, 0) {
  auto values = INPUT_VARIABLE(0)->cast(INT64);

  NDArray *weights = nullptr;

  LongType maxLength = -1;
  LongType minLength = 0;
  LongType maxIndex = values.argMax();
  maxLength = values.e<LongType>(maxIndex) + 1;

  if (block.numI() > 0) {
    minLength = math::sd_max(INT_ARG(0), (LongType) 0L);
    if (block.numI() == 2) maxLength = math::sd_min(maxLength, INT_ARG(1));
  }

  if (block.width() == 2) {  // the second argument is weights
    weights = INPUT_VARIABLE(1);
    if (weights->lengthOf() < 1) {
      weights = NDArrayFactory::create_('c', values.getShapeAsVector(), values.dataType());
      weights->assign(1);
    } else if (weights->isScalar()) {
      auto value = weights->cast(INT64).asVectorT<LongType>();
      weights = NDArrayFactory::create_('c', values.getShapeAsVector(), values.dataType());
      weights->assign(value[0]);
    }

    REQUIRE_TRUE(values.isSameShape(weights), 0, "bincount: the input and weights shapes should be equals");
  } else if (block.width() == 3) {  // the second argument is min and the third is max
    auto min = INPUT_VARIABLE(1);
    auto max = min;
    if (INPUT_VARIABLE(2)->lengthOf() > 0) {
      max = INPUT_VARIABLE(2);
    }
    minLength = min->e<LongType>(0);
    maxLength = max->e<LongType>(0);
  } else if (block.width() > 3) {
    auto min = INPUT_VARIABLE(2);
    auto max = INPUT_VARIABLE(3);
    minLength = min->e<LongType>(0);
    if (INPUT_VARIABLE(2)->lengthOf() > 0) {
      maxLength = max->e<LongType>(0);
    } else
      maxLength = minLength;
    weights = INPUT_VARIABLE(1);
    if (weights->lengthOf() < 1) {
      weights = NDArrayFactory::create_('c', values.getShapeAsVector(), values.dataType());
      weights->assign(1);
    } else if (weights->isScalar()) {
      auto value = weights->asVectorT<LongType>();
      weights = NDArrayFactory::create_('c', values.getShapeAsVector(), values.dataType());
      weights->assign(value[0]);
    }
    REQUIRE_TRUE(values.isSameShape(weights), 0, "bincount: the input and weights shapes should be equals");
  }

  minLength = math::sd_max(minLength, (LongType) 0);
  maxLength = math::sd_min(maxLength, values.e<LongType>(maxIndex) + 1);

  auto result = OUTPUT_VARIABLE(0);
  result->assign(0.0f);

  helpers::adjustWeights(block.launchContext(), &values, weights, result, minLength, maxLength);

  return Status::OK;
}

DECLARE_SHAPE_FN(bincount) {
  auto shapeList = SHAPELIST();
  auto in = INPUT_VARIABLE(0);
  DataType dtype = INT64;
  if (block.width() > 1)
    dtype = ArrayOptions::dataType(inputShape->at(1));
  else if (block.numI() > 2)
    dtype = (DataType)INT_ARG(2);

  LongType maxIndex = in->argMax();
  LongType maxLength = in->e<LongType>(maxIndex) + 1;
  LongType outLength = maxLength;

  if (block.numI() > 0) outLength = math::sd_max(maxLength, INT_ARG(0));

  if (block.numI() > 1) outLength = math::sd_min(outLength, INT_ARG(1));

  if (block.width() == 3) {  // the second argument is min and the third is max
    auto min = INPUT_VARIABLE(1)->e<LongType>(0);
    auto max = min;
    if (INPUT_VARIABLE(2)->lengthOf() > 0) {
      max = INPUT_VARIABLE(2)->e<LongType>(0);
    }

    outLength = math::sd_max(maxLength, min);
    outLength = math::sd_min(outLength, max);
  } else if (block.width() > 3) {
    auto min = INPUT_VARIABLE(2);
    auto max = min;
    if (INPUT_VARIABLE(3)->lengthOf() > 0) {
      max = INPUT_VARIABLE(3);
    }
    outLength = math::sd_max(maxLength, min->e<LongType>(0));
    outLength = math::sd_min(outLength, max->e<LongType>(0));
  }

  auto newshape = ConstantShapeHelper::getInstance().vectorShapeInfo(outLength, dtype);

  shapeList->push_back(newshape);
  return shapeList;
}

}  // namespace ops
}  // namespace sd

#endif
