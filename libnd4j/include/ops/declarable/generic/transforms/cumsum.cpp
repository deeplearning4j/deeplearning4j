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
#if NOT_EXCLUDED(OP_cumsum)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/prefix.h>

namespace sd {
namespace ops {

CONFIGURABLE_OP_IMPL(cumsum, 1, 1, true, 0, 2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const bool exclusive = INT_ARG(0) == 1;
  const bool reverse = INT_ARG(1) == 1;

  REQUIRE_TRUE(input->dataType() == output->dataType(), 0, "CumSum: input and output data types must be equal");

  if (input->isEmpty()) {
    // No-op
    return sd::Status::OK;
  }

  if (block.getIArguments()->size() == 2 && block.width() == 1) {
    // all at once case
    sd::ops::helpers::prefix(block.launchContext(), scalar::Add, input, output, exclusive, reverse);
  } else {
    std::vector<sd::LongType> dims(block.numI() - 2);

    if (block.width() == 1) {
      for (size_t e = 0; e < block.numI() - 2; e++) dims[e] = INT_ARG(e + 2);
    } else {
      auto ax = INPUT_VARIABLE(1);
      dims = ax->template asVectorT<sd::LongType>();
    }

    for (size_t e = 0; e < dims.size(); e++)
      if (dims[e] < 0) dims[e] += input->rankOf();

    sd::ops::helpers::prefix(block.launchContext(), scalar::Add, input, output, dims, exclusive, reverse);
  }

  return sd::Status::OK;
}
DECLARE_TYPES(cumsum) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS, ALL_INTS})
      ->setAllowedInputTypes(1, {ALL_FLOATS,ALL_INTS})
      ->setAllowedOutputTypes({ALL_FLOATS})
      ->setSameMode(false);
}

CUSTOM_OP_IMPL(cumsum_bp, 2, -1, true, 0, 2) {
  auto input = INPUT_VARIABLE(0);
  auto axis = block.width() == 3 ? INPUT_VARIABLE(1) : nullptr;
  auto gradOut = block.width() == 3 ? INPUT_VARIABLE(2) : INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);
  const bool exclusive = INT_ARG(0) == 1;
  const bool reverse = INT_ARG(1) == 1;

  std::vector<sd::LongType> dims;

  if (block.width() > 2) {
    dims = axis->template asVectorT<sd::LongType>();
    float one = 1.f;
    OUTPUT_VARIABLE(1)->assign(one);
  } else if (int newSize = (block.numI() - 2)) {
    dims.resize(newSize);

    for (int e = 0; e < newSize; e++) dims[e] = INT_ARG(e + 2);
  }
  if (!exclusive && !reverse) {
    if (dims.size())
      sd::ops::helpers::prefix(block.launchContext(), scalar::Add, gradOut, output, dims, false, true);
    else
      sd::ops::helpers::prefix(block.launchContext(), scalar::Add, gradOut, output, false, true);

  } else if (!exclusive && reverse) {
    if (dims.size())
      sd::ops::helpers::prefix(block.launchContext(), scalar::Add, gradOut, output, dims, false, false);
    else
      sd::ops::helpers::prefix(block.launchContext(), scalar::Add, gradOut, output, false, false);
  } else if (exclusive && !reverse) {
    if (dims.size())
      sd::ops::helpers::prefix(block.launchContext(), scalar::Add, gradOut, output, dims, true, true);
    else
      sd::ops::helpers::prefix(block.launchContext(), scalar::Add, gradOut, output, true, true);
  } else {
    if (dims.size())
      sd::ops::helpers::prefix(block.launchContext(), scalar::Add, gradOut, output, dims, true, false);
    else
      sd::ops::helpers::prefix(block.launchContext(), scalar::Add, gradOut, output, true, false);
  }

  return sd::Status::OK;
}
DECLARE_TYPES(cumsum_bp) {
  getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS, ALL_INTS});
  getOpDescriptor()->setAllowedInputTypes(1, {ALL_FLOATS, ALL_INTS});  // axes can be set as the second param
  getOpDescriptor()->setAllowedInputTypes(2, {ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes(0, {ALL_FLOATS});
}

DECLARE_SHAPE_FN(cumsum_bp) {
  auto inp = inputShape->at(0);
  sd::LongType *newShapeX = nullptr;
  COPY_SHAPE(inp, newShapeX);

  if (block.width() == 2) {
    auto result = CONSTANT(newShapeX);
    delete[] newShapeX;
    return SHAPELIST(result);
  } else {
    sd::LongType *newShapeA = nullptr;
    COPY_SHAPE(inputShape->at(1), newShapeA);

    auto resultX = CONSTANT(newShapeX);
    auto resultA = CONSTANT(newShapeA);
    delete[] newShapeX;
    delete[] newShapeA;
    return SHAPELIST(resultX, resultA);
  }
}
}  // namespace ops
}  // namespace sd

#endif
