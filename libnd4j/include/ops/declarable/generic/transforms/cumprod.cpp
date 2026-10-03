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
#if NOT_EXCLUDED(OP_cumprod)

#include <ops/declarable/headers/transforms.h>
#include <ops/declarable/helpers/prefix.h>

namespace sd {
namespace ops {
CONFIGURABLE_OP_IMPL(cumprod, 1, 1, true, 0, 2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->dataType() == output->dataType(), 0, "CumSum: input and output data types must be equal");

  if (input->isEmpty()) {
    // No-op
    return sd::Status::OK;
  }

  const bool exclusive = INT_ARG(0) == 1;
  const bool reverse = INT_ARG(1) == 1;

  if (block.getIArguments()->size() == 2 && block.width() == 1) {
    // all at once case
    sd::ops::helpers::prefix(block.launchContext(), scalar::Multiply, input, output, exclusive, reverse);
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

    sd::ops::helpers::prefix(block.launchContext(), scalar::Multiply, input, output, dims, exclusive, reverse);
  }

  return sd::Status::OK;
}

DECLARE_TYPES(cumprod) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, sd::DataType::ANY)
      ->setAllowedInputTypes(1, {ALL_INTS})
      ->setAllowedOutputTypes({ALL_FLOATS})
      ->setSameMode(true);
  getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING);
}

DECLARE_TYPES(cumprod_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, sd::DataType::ANY)
      ->setAllowedInputTypes(1, {ALL_INTS, ALL_FLOATS})  // there is a case when axes given as IArgs
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS})
      ->setSameMode(true);
}

// With y = cumprod(x) scanned along some direction (strictly before each element when exclusive), output i holds
// every x_k at or before i in that direction (strictly before i when exclusive), and dy_i/dx_k = y_i / x_k for those
// k. So
//   dL/dx_k = (sum of g_i * y_i over the outputs i that hold x_k) / x_k,
// where the outputs that hold x_k are the ones from k on in the scan direction (after k when exclusive): the sum is a
// cumulative sum of g * y in the opposite direction, exclusive when the forward scan was, divided by x.
// TensorFlow's Cumprod gradient divides the same way, so like it this is not defined for an x_k of zero.
CUSTOM_OP_IMPL(cumprod_bp, 2, 1, false, 0, 2) {
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

  for (size_t e = 0; e < dims.size(); e++)
    if (dims[e] < 0) dims[e] += input->rankOf();

  if (input->isEmpty()) {
    // No-op
    return sd::Status::OK;
  }

  // Without axes the forward op scans the whole array as one flat sequence, and so must its gradient: the scan
  // over a list of axes has nothing to scan when the list is empty.
  auto scan = [&](scalar::Ops op, NDArray* x, NDArray* z, const bool exclusiveScan, const bool reverseScan) {
    if (dims.empty())
      sd::ops::helpers::prefix(block.launchContext(), op, x, z, exclusiveScan, reverseScan);
    else
      sd::ops::helpers::prefix(block.launchContext(), op, x, z, dims, exclusiveScan, reverseScan);
  };

  // y = cumprod(x), scanned as the forward op scanned it
  scan(scalar::Multiply, input, output, exclusive, reverse);

  // g * y, accumulated against the scan direction and divided by x. A loss variable that is not itself a scalar
  // sends back a scalar gradient, the same at every output. `weighted` is a copy of y so that it lays out its axes
  // the way the output does: the scans pair the sequences of both arrays by position.
  NDArray *weighted = output->dup();
  if (gradOut->lengthOf() == 1 && output->lengthOf() > 1)
    output->applyScalarArr(scalar::Multiply, gradOut, weighted);
  else
    gradOut->applyPairwiseTransform(pairwise::Multiply, output, weighted);
  scan(scalar::Add, weighted, output, exclusive, !reverse);
  output->applyPairwiseTransform(pairwise::Divide, input, output);

  delete weighted;

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(cumprod_bp) {
  auto inp = inputShape->at(0);
  if (block.width() == 2) {
    return SHAPELIST(CONSTANT(inp));
  } else {
    return SHAPELIST(CONSTANT(inp), CONSTANT(inputShape->at(1)));
  }
}
}  // namespace ops
}  // namespace sd

#endif
