/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// Created by raver119 on 01.11.2017.
// Modified by GS <sgazeos@gmail.com> 4/5/2018

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_argmax)

#include <helpers/ConstantTadHelper.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/axis.h>
#include <ops/declarable/helpers/reductions.h>

namespace sd {
namespace ops {
DECLARE_TYPES(argmax) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS, ALL_INTS})->setAllowedOutputTypes({ANY});
}

CUSTOM_OP_IMPL(argmax, 1, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  if (output->isEmpty() || output->lengthOf() < 1) return Status::OK;

  auto axis = *block.getIArguments();

  // axis might be dynamic (i.e. tf mode)
  if (block.width() > 1 && axis.size() == 0) {
    auto axisVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(input->rankOf(), axisVector, axis);
    helpers::argMax(*input, *output, axis);
  } else {
    helpers::argMax(*input, *output, axis);
  }

  STORE_RESULT(output);

  return Status::OK;
}

DECLARE_SHAPE_FN(argmax) {
  auto firstInputShape = inputShape->at(0);
  if(shape::isScalar(firstInputShape)) {
    return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(DataType::INT64));
  }
  std::vector<LongType> dims;

  if (block.width() == 1) {
    dims = *block.getIArguments();
  } else {
    auto y = INPUT_VARIABLE(1)->cast(INT64);
    dims = y.template asVectorT<LongType>();
  }

  auto keepDims = block.numB() ? B_ARG(0) : false;
  auto dtype = block.numD() ? D_ARG(0) : INT64;

  // we're resolving negative axis here
  helpers::adjustAxis(shape::rank(inputShape->at(0)), dims);


  for (auto d : dims) {
    // we have special case here
    if (d == DataTypeUtils::max<int>()) continue;

    REQUIRE_TRUE(d < shape::rank(firstInputShape), 0, "ArgMax: axis can't be above rank")
    REQUIRE_TRUE(firstInputShape[d + 1] != 0, 0, "ArgMax: you can't reduce along axis with 0 in shape");
  }

  // special case - output is scalar
  if (dims.empty() || (dims.size() == 1 && dims.at(0) == DataTypeUtils::max<int>())) {
    return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(dtype));
  }

  auto ret = ShapeUtils::evalReduceShapeInfo('c', &dims, firstInputShape, dtype, keepDims, false, block.getWorkspace());
  return SHAPELIST(ret);
}
}  // namespace ops
}  // namespace sd

#endif
