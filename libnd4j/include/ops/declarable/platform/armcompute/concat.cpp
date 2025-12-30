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

#include <numeric>

#include "armcomputeUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(concat, ENGINE_CPU) {
  REQUIRE_TRUE(block.width() > 0, 0, "CONCAT ARMCOMPUTE op: No input arrays were provided");

  const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);
  const int numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();

  // first of all take into account possible presence of empty arrays
  std::vector<NDArray*> nonEmptyArrs;
  std::vector<sd::LongType> arrsToDelete;
  int index = 0;
  bool allOfSameType = true;
  auto typeOfFirstArr = block.width() > 0 ? INPUT_VARIABLE(0)->dataType() : block.dataType();

  for (int i = 0; i < numOfInArrs; ++i) {
    auto input = INPUT_VARIABLE(i);
    if (!input->isEmpty()) {
      allOfSameType &= (typeOfFirstArr == input->dataType());

      if (input->rankOf() == 0) {
        std::vector<sd::LongType> dim = {1};
        auto vec = new NDArray('c', dim, input->dataType(), block.launchContext());
        vec->assign(input);
        nonEmptyArrs.push_back(vec);
        arrsToDelete.push_back(index);
      } else {
        nonEmptyArrs.push_back(input);
      }
      ++index;
    }
  }

  const int numOfNonEmptyArrs = nonEmptyArrs.size();

  if (numOfNonEmptyArrs == 0) {
    REQUIRE_TRUE(OUTPUT_VARIABLE(0)->isEmpty(), 0,
                 "CONCAT ARMCOMPUTE op: If all input variables are empty, output must be empty");
    return sd::Status::OK;
  }

  const int rank = nonEmptyArrs[0]->rankOf();
  int axis = isAxisInLastArr ? INPUT_VARIABLE(block.width() - 1)->e<int>(0) : INT_ARG(0);
  if (axis < 0) {
    axis += rank;
  }

  // Input validation
  REQUIRE_TRUE(allOfSameType, 0, "CONCAT ARMCOMPUTE op: all of input arrays must have same type !");
  REQUIRE_TRUE(nonEmptyArrs[0]->dataType() == OUTPUT_VARIABLE(0)->dataType(), 0,
               "CONCAT ARMCOMPUTE op: output array should have the same type as inputs arrays !");
  REQUIRE_TRUE(0 <= axis && (axis < rank || (axis == 0 && rank == 0)), 0,
               "CONCAT ARMCOMPUTE op: input axis must be in range [0, %i], but got %i instead!", rank - 1, axis);

  for (int i = 1; i < numOfNonEmptyArrs; ++i)
    REQUIRE_TRUE(nonEmptyArrs[i]->rankOf() == rank, 0, "CONCAT ARMCOMPUTE op: all input arrays must have the same rank !");

  for (int i = 1; i < numOfNonEmptyArrs; ++i) {
    for (int dim = 0; dim < rank; ++dim)
      if (dim != axis)
        REQUIRE_TRUE(nonEmptyArrs[i]->sizeAt(dim) == nonEmptyArrs[0]->sizeAt(dim), 0,
                     "CONCAT ARMCOMPUTE op: all input arrays must have the same dimensions (except those on input axis) !");
  }

  auto output = OUTPUT_VARIABLE(0);

  if (numOfNonEmptyArrs == 1) {
    output->assign(nonEmptyArrs[0]);
  } else {
    // ARM Compute axis is in reverse order (from the end)
    int armAxis = rank - 1 - axis;

    // Create ARM tensors for inputs
    std::vector<Arm_Tensor> inputTensors(numOfNonEmptyArrs);
    std::vector<arm_compute::ITensor*> inputTensorPtrs(numOfNonEmptyArrs);
    std::vector<NDArray*> inputNdsForCopy;

    for (int i = 0; i < numOfNonEmptyArrs; ++i) {
      auto inInfo = getArmTensorInfo(*nonEmptyArrs[i], Arm_DataLayout::UNKNOWN);
      inputTensors[i].allocator()->init(inInfo);
      inputTensorPtrs[i] = &inputTensors[i];
    }

    // Create output tensor
    Arm_Tensor outputTensor;
    auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);
    outputTensor.allocator()->init(outInfo);

    // Configure concatenation
    arm_compute::NEConcatenateLayer concat;
    concat.configure(inputTensorPtrs, &outputTensor, armAxis);

    // Allocate and import memory for inputs
    for (int i = 0; i < numOfNonEmptyArrs; ++i) {
      if (!nonEmptyArrs[i]->hasPaddedBuffer() && !inputTensors[i].info()->has_padding()) {
        inputTensors[i].allocator()->import_memory(nonEmptyArrs[i]->buffer());
      } else {
        inputTensors[i].allocator()->allocate();
        copyToTensor(*nonEmptyArrs[i], inputTensors[i]);
      }
    }

    // Allocate and import memory for output
    bool copyOutput = false;
    if (!output->hasPaddedBuffer() && !outputTensor.info()->has_padding()) {
      outputTensor.allocator()->import_memory(output->buffer());
    } else {
      outputTensor.allocator()->allocate();
      copyOutput = true;
    }

    // Run concatenation
    concat.run();

    // Copy output if needed
    if (copyOutput) {
      copyFromTensor(outputTensor, *output);
    }
  }

  // Clean up dynamically allocated arrays
  for (auto idx : arrsToDelete) {
    delete nonEmptyArrs[idx];
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(concat, ENGINE_CPU) {
  auto z = OUTPUT_VARIABLE(0);

  const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);
  const int numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();

  Requirements req("ARMCOMPUTE CONCAT OP");
  req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(z->rankOf(), RANK_MSG_OUTPUT), arm_compute::MAX_DIMS) &&
      req.expectEq(makeInfoVariable(z->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(z->stridesOf()[z->rankOf() - 1], "output#lastStride"), 1);

  // Check all inputs are FLOAT32 and contiguous
  for (int i = 0; i < numOfInArrs; ++i) {
    auto input = INPUT_VARIABLE(i);
    if (!input->isEmpty()) {
      req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
          req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
          req.expectEq(makeInfoVariable(input->stridesOf()[input->rankOf() - 1], "input#lastStride"), 1);
    }
  }

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
