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

namespace {

constexpr int kStridedSliceMaskArguments = 5;
constexpr int kArmComputeStridedSliceMaxRank = 4;

int reverseMaskBits(int mask, int rank) {
  int reversed = 0;
  for (int i = 0; i < rank; ++i) {
    if ((mask & (1 << i)) != 0) reversed |= 1 << (rank - 1 - i);
  }
  return reversed;
}

void makeArmSliceCoordinates(const std::vector<int>& begin, const std::vector<int>& end,
                             const std::vector<int>& strides, arm_compute::Coordinates& starts,
                             arm_compute::Coordinates& ends, arm_compute::BiStrides& armStrides) {
  const int rank = static_cast<int>(begin.size());
  for (int i = rank - 1; i >= 0; --i) {
    const int armDimension = rank - 1 - i;
    starts.set(armDimension, begin[i]);
    ends.set(armDimension, end[i]);
    armStrides.set(armDimension, strides[i]);
  }
}

}  // namespace

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(strided_slice, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  int rank = input->rankOf();

  const int beginMask = INT_ARG(0);
  const int endMask = INT_ARG(2);
  const int shrinkAxisMask = INT_ARG(4);

  // Get begin, end, strides. Static coordinates follow the five mask arguments.
  std::vector<int> begin(rank), end(rank), strides(rank);

  if (block.width() > 3) {
    auto beginArr = INPUT_VARIABLE(1);
    auto endArr = INPUT_VARIABLE(2);
    auto stridesArr = INPUT_VARIABLE(3);
    for (int i = 0; i < rank; i++) {
      begin[i] = beginArr->e<int>(i);
      end[i] = endArr->e<int>(i);
      strides[i] = stridesArr->e<int>(i);
    }
  } else {
    for (int i = 0; i < rank; i++) {
      begin[i] = INT_ARG(kStridedSliceMaskArguments + i);
      end[i] = INT_ARG(kStridedSliceMaskArguments + rank + i);
      strides[i] = INT_ARG(kStridedSliceMaskArguments + 2 * rank + i);
    }
  }

  // Create ARM Compute coordinates (reversed order)
  arm_compute::Coordinates starts, ends;
  arm_compute::BiStrides armStrides;
  makeArmSliceCoordinates(begin, end, strides, starts, ends, armStrides);

  const int armBeginMask = reverseMaskBits(beginMask, rank);
  const int armEndMask = reverseMaskBits(endMask, rank);
  const int armShrinkAxisMask = reverseMaskBits(shrinkAxisMask, rank);

  auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);

  Arm_Tensor inTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  outTensor.allocator()->init(outInfo);

  arm_compute::NEStridedSlice stridedSlice;
  stridedSlice.configure(&inTensor, &outTensor, starts, ends, armStrides, armBeginMask, armEndMask,
                         armShrinkAxisMask);

  if (!input->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input, inTensor);
  }

  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  stridedSlice.run();

  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(strided_slice, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const int rank = input->rankOf();
  const int outputRank = output->rankOf();
  const int beginMask = INT_ARG(0);
  const int ellipsisMask = INT_ARG(1);
  const int endMask = INT_ARG(2);
  const int newAxisMask = INT_ARG(3);
  const int shrinkAxisMask = INT_ARG(4);

  const int validMaskBits = rank > 0 && rank <= kArmComputeStridedSliceMaxRank ? (1 << rank) - 1 : 0;
  const bool supportedMasks = ellipsisMask == 0 && newAxisMask == 0 &&
                              ((beginMask | endMask | shrinkAxisMask) & ~validMaskBits) == 0;
  const bool dynamicSpec = block.width() > 3;
  const bool staticSpec = block.width() == 1;
  bool completeSpec = false;
  if (dynamicSpec) {
    completeSpec = INPUT_VARIABLE(1)->lengthOf() == rank && INPUT_VARIABLE(2)->lengthOf() == rank &&
                   INPUT_VARIABLE(3)->lengthOf() == rank;
  } else if (staticSpec) {
    completeSpec = block.getIArguments()->size() ==
                   static_cast<size_t>(kStridedSliceMaskArguments + 3 * rank);
  }

  const bool inputLastStrideIsOne = rank > 0 && input->stridesOf()[rank - 1] == 1;
  const bool outputLastStrideIsOne = outputRank > 0 && output->stridesOf()[outputRank - 1] == 1;

  Requirements req("ARMCOMPUTE STRIDED_SLICE OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(rank, RANK_MSG_INPUT), kArmComputeStridedSliceMaxRank) &&
      req.expectGreater(makeInfoVariable(rank, RANK_MSG_INPUT), 0) &&
      req.expectGreater(makeInfoVariable(outputRank, RANK_MSG_OUTPUT), 0) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectTrue(makeInfoVariable(inputLastStrideIsOne, "input#lastStrideIsOne")) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectTrue(makeInfoVariable(outputLastStrideIsOne, "output#lastStrideIsOne")) &&
      req.expectFalse(makeInfoVariable(output->isEmpty(), "output#empty")) &&
      req.expectTrue(makeInfoVariable(supportedMasks, "supportedMasks")) &&
      req.expectTrue(makeInfoVariable(completeSpec, "completeSliceSpec"));

  if (req) {
    std::vector<int> begin(rank), end(rank), strides(rank);
    if (dynamicSpec) {
      auto beginArr = INPUT_VARIABLE(1);
      auto endArr = INPUT_VARIABLE(2);
      auto stridesArr = INPUT_VARIABLE(3);
      for (int i = 0; i < rank; ++i) {
        begin[i] = beginArr->e<int>(i);
        end[i] = endArr->e<int>(i);
        strides[i] = stridesArr->e<int>(i);
      }
    } else {
      for (int i = 0; i < rank; ++i) {
        begin[i] = INT_ARG(kStridedSliceMaskArguments + i);
        end[i] = INT_ARG(kStridedSliceMaskArguments + rank + i);
        strides[i] = INT_ARG(kStridedSliceMaskArguments + 2 * rank + i);
      }
    }

    arm_compute::Coordinates starts, ends;
    arm_compute::BiStrides armStrides;
    makeArmSliceCoordinates(begin, end, strides, starts, ends, armStrides);

    auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);
    auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);
    auto status = arm_compute::NEStridedSlice::validate(
        &inInfo, &outInfo, starts, ends, armStrides, reverseMaskBits(beginMask, rank),
        reverseMaskBits(endMask, rank), reverseMaskBits(shrinkAxisMask, rank));
    req.expectTrue(makeInfoVariable(static_cast<bool>(status), "armComputeValidation"));
  }
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
