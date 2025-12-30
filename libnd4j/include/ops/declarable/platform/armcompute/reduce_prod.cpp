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
PLATFORM_IMPL(reduce_prod, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Get dimensions to reduce
  std::vector<int> dimensions;
  if (block.width() > 1) {
    auto dims = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < dims->lengthOf(); i++) {
      dimensions.push_back(dims->e<int>(i));
    }
  } else if (block.numI() > 0) {
    for (int i = 0; i < block.numI(); i++) {
      dimensions.push_back(INT_ARG(i));
    }
  }

  // Handle negative dimensions
  for (auto& d : dimensions) {
    if (d < 0) d += input->rankOf();
  }

  // Get keepDims
  bool keepDims = false;
  if (block.numB() > 0) {
    keepDims = B_ARG(0);
  }

  // Create tensor info
  auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);

  Arm_Tensor inTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  outTensor.allocator()->init(outInfo);

  // For single axis reduction
  if (dimensions.size() == 1) {
    unsigned int armAxis = input->rankOf() - 1 - dimensions[0];

    arm_compute::NEReductionOperation reduce;
    reduce.configure(&inTensor, &outTensor, armAxis, arm_compute::ReductionOperation::PROD, keepDims);

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

    reduce.run();

    if (copyOutput) {
      copyFromTensor(outTensor, *output);
    }
  } else {
    if (!input->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
      inTensor.allocator()->import_memory(input->buffer());
    } else {
      inTensor.allocator()->allocate();
      copyToTensor(*input, inTensor);
    }

    std::sort(dimensions.begin(), dimensions.end(), std::greater<int>());

    Arm_Tensor* currentIn = &inTensor;
    std::vector<std::unique_ptr<Arm_Tensor>> tempTensors;

    for (size_t i = 0; i < dimensions.size(); i++) {
      unsigned int armAxis = currentIn->info()->num_dimensions() - 1 - dimensions[i];

      Arm_Tensor* currentOut;
      if (i == dimensions.size() - 1) {
        currentOut = &outTensor;
        if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
          outTensor.allocator()->import_memory(output->buffer());
        } else {
          outTensor.allocator()->allocate();
        }
      } else {
        tempTensors.push_back(std::make_unique<Arm_Tensor>());
        currentOut = tempTensors.back().get();
      }

      arm_compute::NEReductionOperation reduce;
      reduce.configure(currentIn, currentOut, armAxis, arm_compute::ReductionOperation::PROD, keepDims);

      if (i < dimensions.size() - 1) {
        currentOut->allocator()->allocate();
      }

      reduce.run();
      currentIn = currentOut;
    }

    if (output->hasPaddedBuffer() || outTensor.info()->has_padding()) {
      copyFromTensor(outTensor, *output);
    }
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(reduce_prod, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE REDUCE_PROD OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(input->stridesOf()[input->rankOf() - 1], "input#lastStride"), 1) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->stridesOf()[output->rankOf() - 1], "output#lastStride"), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
