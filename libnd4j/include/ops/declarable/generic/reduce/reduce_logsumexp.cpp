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
// Created by george@skymind.io on 11/13/2018.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/axis.h>
namespace sd {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_logsumexp)

CUSTOM_OP_IMPL(reduce_logsumexp, -1, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  std::vector<sd::LongType> axes;  // = *block.getIArguments();
  if (block.width() > 1) {
    auto axisVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(input->rankOf(), axisVector, axes);
  } else if (block.getIArguments()->size() > 0) {
    axes = *block.getIArguments();
  }

  for (const auto& item : axes)
    REQUIRE_TRUE(
        item >= -input->shapeInfo()[0] && item < input->shapeInfo()[0], 0,
        "REDUCE_LOGSUMEXP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !",
        input->rankOf(), input->rankOf(), item);

  const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
  sd::LongType maxI = input->argMax();
  auto maxVals = input->e(maxI);
  // void* whereMax = (void*)();
  auto internal = (*input);
  internal -= maxVals;
  internal.applyTransform(transform::Exp, &internal);
  internal.reduceAlongDimension(reduce::Sum, output, &axes, keepDims, false);  //, (void*)&maxVals);
  output->applyTransform(transform::Log, output);
  (*output) += maxVals;
  return sd::Status::OK;
}
DECLARE_TYPES(reduce_logsumexp) {
  getOpDescriptor()->setAllowedInputTypes({ALL_INTS, ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS});
}
DECLARE_SHAPE_FN(reduce_logsumexp) {
  const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
  auto input = INPUT_VARIABLE(0);

  std::vector<sd::LongType> axes;  // = *block.getIArguments();
  if (block.width() > 1) {
    auto axisVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(input->rankOf(), axisVector, axes);
  } else if (block.getIArguments()->size() > 0) {
    axes = *block.getIArguments();
  }

  auto outShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(inputShape->at(0)), &axes, inputShape->at(0),
                                                      keepDims, false, block.getWorkspace());

  return SHAPELIST(outShapeInfo);
}
#endif
}  // namespace ops
}  // namespace sd
