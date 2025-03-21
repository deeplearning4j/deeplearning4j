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
// Created by sgazeos@gmail.com on 26.01.2018.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_moments)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/axis.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(moments, 1, 2, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto means = OUTPUT_VARIABLE(0);
  auto variances = OUTPUT_VARIABLE(1);

  std::vector<LongType> axis = *block.getIArguments();
  const bool keepDims = block.getBArguments()->size() > 0 ? (bool)B_ARG(0) : false;
  reduce_variance varianceOp;

  // axis might be dynamic (i.e. tf mode)
  if (block.width() > 1) {
    auto axisVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(input->rankOf(), axisVector, axis);
    varianceOp.execute({input, axisVector}, {variances}, {}, {}, {keepDims}, {}, false);
  } else {
    std::vector<LongType>& dims = axis;
    std::vector<LongType> axes;
    for (size_t i = 0; i < dims.size(); i++) {
      axes.push_back(dims[i]);
    }

    varianceOp.execute({input}, {variances}, {}, axes, {keepDims}, {}, false);
  }

  input->reduceAlongDimension(reduce::Mean, means, &axis, keepDims);

  return Status::OK;
}

DECLARE_SHAPE_FN(moments) {
  auto axis = *block.getIArguments();
  auto input = INPUT_VARIABLE(0);

  // axis might be dynamic (i.e. tf mode)
  if (block.width() > 1 && axis.size() == 0) {
    auto axisVector = INPUT_VARIABLE(1);

    for (int e = 0; e < axisVector->lengthOf(); e++) {
      int ca = axisVector->e<int>(e);
      if (ca < 0) ca += input->rankOf();

      axis.emplace_back(ca);
    }
  }
  const bool keepDims = block.getBArguments()->size() > 0 ? (bool)B_ARG(0) : false;

  auto meanShape = ShapeUtils::evalReduceShapeInfo('c', &axis, *input, keepDims, false, block.workspace());
  auto varianceShape = ShapeUtils::evalReduceShapeInfo('c', &axis, *input, keepDims, false, block.workspace());
  return SHAPELIST(meanShape, varianceShape);
}

DECLARE_TYPES(moments) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}
}  // namespace ops

}  // namespace sd

#endif
