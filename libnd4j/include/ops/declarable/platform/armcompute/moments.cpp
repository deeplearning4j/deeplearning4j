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
PLATFORM_IMPL(moments, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto meanOutput = OUTPUT_VARIABLE(0);
  auto varianceOutput = OUTPUT_VARIABLE(1);

  // Get dimensions to compute moments over
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

  // Convert dimensions to ARM Compute coordinates
  arm_compute::Coordinates reduction_axis;
  for (int d : dimensions) {
    reduction_axis.set(reduction_axis.num_dimensions(), input->rankOf() - 1 - d);
  }

  auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);
  auto meanInfo = getArmTensorInfo(*meanOutput, Arm_DataLayout::UNKNOWN);
  auto varInfo = getArmTensorInfo(*varianceOutput, Arm_DataLayout::UNKNOWN);

  Arm_Tensor inTensor, meanTensor, varTensor;
  inTensor.allocator()->init(inInfo);
  meanTensor.allocator()->init(meanInfo);
  varTensor.allocator()->init(varInfo);

  // Compute mean using NEReduceMean
  arm_compute::NEReduceMean reduceMean;
  reduceMean.configure(&inTensor, reduction_axis, true, &meanTensor);

  if (!input->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input, inTensor);
  }

  bool copyMean = false;
  if (!meanOutput->hasPaddedBuffer() && !meanTensor.info()->has_padding()) {
    meanTensor.allocator()->import_memory(meanOutput->buffer());
  } else {
    meanTensor.allocator()->allocate();
    copyMean = true;
  }

  reduceMean.run();

  if (copyMean) {
    copyFromTensor(meanTensor, *meanOutput);
  }

  // Compute variance: E[x^2] - E[x]^2
  // First compute squared input
  Arm_Tensor squaredTensor;
  squaredTensor.allocator()->init(inInfo);
  squaredTensor.allocator()->allocate();

  arm_compute::NEPixelWiseMultiplication square;
  square.configure(&inTensor, &inTensor, &squaredTensor, 1.0f, arm_compute::ConvertPolicy::SATURATE,
                   arm_compute::RoundingPolicy::TO_NEAREST_EVEN);
  square.run();

  // Compute mean of squared
  Arm_Tensor meanSquaredTensor;
  meanSquaredTensor.allocator()->init(varInfo);
  meanSquaredTensor.allocator()->allocate();

  arm_compute::NEReduceMean reduceMeanSquared;
  reduceMeanSquared.configure(&squaredTensor, reduction_axis, true, &meanSquaredTensor);
  reduceMeanSquared.run();

  // Compute mean^2
  Arm_Tensor squaredMeanTensor;
  squaredMeanTensor.allocator()->init(meanInfo);
  squaredMeanTensor.allocator()->allocate();

  arm_compute::NEPixelWiseMultiplication squareMean;
  squareMean.configure(&meanTensor, &meanTensor, &squaredMeanTensor, 1.0f, arm_compute::ConvertPolicy::SATURATE,
                       arm_compute::RoundingPolicy::TO_NEAREST_EVEN);
  squareMean.run();

  // Variance = E[x^2] - E[x]^2
  bool copyVar = false;
  if (!varianceOutput->hasPaddedBuffer() && !varTensor.info()->has_padding()) {
    varTensor.allocator()->import_memory(varianceOutput->buffer());
  } else {
    varTensor.allocator()->allocate();
    copyVar = true;
  }

  arm_compute::NEArithmeticSubtraction subtract;
  subtract.configure(&meanSquaredTensor, &squaredMeanTensor, &varTensor, arm_compute::ConvertPolicy::SATURATE);
  subtract.run();

  if (copyVar) {
    copyFromTensor(varTensor, *varianceOutput);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(moments, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto meanOutput = OUTPUT_VARIABLE(0);
  auto varianceOutput = OUTPUT_VARIABLE(1);

  Requirements req("ARMCOMPUTE MOMENTS OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(meanOutput->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(varianceOutput->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(input->stridesOf()[input->rankOf() - 1], "input#lastStride"), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
