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

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "armcomputeUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// Cumulative sum along an axis
PLATFORM_IMPL(cumsum, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  bool exclusive = INT_ARG(0) != 0;
  bool reverse = INT_ARG(1) != 0;

  // Get axis - can be from input or int arg
  int axis = block.width() > 1 ? INPUT_VARIABLE(1)->e<int>(0) : INT_ARG(2);

  // Handle negative axis
  if (axis < 0) axis += input->rankOf();

  auto inPtr = input->bufferAsT<float>();
  auto outPtr = output->bufferAsT<float>();

  // Get dimensions
  auto shape = input->shapeOf();
  int rank = input->rankOf();

  // Calculate strides for iteration
  std::vector<sd::LongType> strides(rank);
  strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  sd::LongType axisSize = shape[axis];
  sd::LongType axisStride = strides[axis];

  // Total elements divided by axis size gives number of "lines" along axis
  sd::LongType numLines = input->lengthOf() / axisSize;

  // For each line along the axis
  for (sd::LongType lineIdx = 0; lineIdx < numLines; lineIdx++) {
    // Calculate base index for this line
    sd::LongType baseIdx = 0;
    sd::LongType temp = lineIdx;
    for (int d = 0; d < rank; d++) {
      if (d != axis) {
        sd::LongType dimSize = shape[d];
        sd::LongType coord = temp % dimSize;
        temp /= dimSize;
        baseIdx += coord * strides[d];
      }
    }

    // Compute cumsum along this line
    if (!reverse) {
      float sum = 0.0f;
      for (sd::LongType i = 0; i < axisSize; i++) {
        sd::LongType idx = baseIdx + i * axisStride;
        if (exclusive) {
          outPtr[idx] = sum;
          sum += inPtr[idx];
        } else {
          sum += inPtr[idx];
          outPtr[idx] = sum;
        }
      }
    } else {
      float sum = 0.0f;
      for (sd::LongType i = axisSize - 1; i >= 0; i--) {
        sd::LongType idx = baseIdx + i * axisStride;
        if (exclusive) {
          outPtr[idx] = sum;
          sum += inPtr[idx];
        } else {
          sum += inPtr[idx];
          outPtr[idx] = sum;
        }
      }
    }
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(cumsum, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE CUMSUM OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c');
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
