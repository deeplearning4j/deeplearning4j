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
#include <ops/declarable/helpers/convolutions.h>
#include <system/platform_boilerplate.h>

#include "armcomputeUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(batchnorm, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);     // 2D:nc, 4D:nchw/nhwc, 5D:ncdhw/ndhwc
  auto mean = INPUT_VARIABLE(1);      // [c]
  auto variance = INPUT_VARIABLE(2);  // [c]
  NDArray* gamma = nullptr;           // [c]
  NDArray* beta = nullptr;            // [c]

  auto output = OUTPUT_VARIABLE(0);  // same shape as input

  const bool applyScale = (bool)INT_ARG(0);
  const bool applyOffset = (bool)INT_ARG(1);
  const float epsilon = static_cast<float>(T_ARG(0));

  if (applyScale) gamma = INPUT_VARIABLE(3);
  if (applyOffset) beta = INPUT_VARIABLE(3 + (int)applyScale);

  const int numOfIntArgs = block.getIArguments()->size();
  const int inRank = input->rankOf();

  // get axes args to normalize input array over
  std::vector<sd::LongType> axes;
  if (numOfIntArgs > 2)
    for (int i = 2; i < numOfIntArgs; ++i) axes.push_back(INT_ARG(i));
  else
    axes.push_back(inRank - 1);  // default dimension to reduce along is last dimension

  REQUIRE_TRUE(axes.size() == 1, 0,
               "BATCHNORM ARMCOMPUTE op: ARM Compute library supports only one axis which represents channel dimension, "
               "but got %i axes instead!",
               axes.size());
  REQUIRE_TRUE(inRank == 4, 0,
               "BATCHNORM ARMCOMPUTE op: ARM Compute library supports only rank 4 input (NCHW/NHWC), but got %i instead!",
               inRank);

  // Determine data layout based on axis
  const bool isNCHW = (axes[0] == 1);
  auto dataLayout = isNCHW ? arm_compute::DataLayout::NCHW : arm_compute::DataLayout::NHWC;

  // Get input tensor info
  auto inInfo = getArmTensorInfo(*input, dataLayout);
  auto outInfo = getArmTensorInfo(*output, dataLayout);
  auto meanInfo = getArmTensorInfo(*mean, Arm_DataLayout::UNKNOWN);
  auto varInfo = getArmTensorInfo(*variance, Arm_DataLayout::UNKNOWN);

  Arm_Tensor inTensor;
  Arm_Tensor outTensor;
  Arm_Tensor meanTensor;
  Arm_Tensor varTensor;
  Arm_Tensor gammaTensor;
  Arm_Tensor betaTensor;

  // Initialize tensors
  inTensor.allocator()->init(inInfo);
  outTensor.allocator()->init(outInfo);
  meanTensor.allocator()->init(meanInfo);
  varTensor.allocator()->init(varInfo);

  Arm_Tensor* gammaPtr = nullptr;
  Arm_Tensor* betaPtr = nullptr;

  if (gamma != nullptr) {
    auto gammaInfo = getArmTensorInfo(*gamma, Arm_DataLayout::UNKNOWN);
    gammaTensor.allocator()->init(gammaInfo);
    gammaPtr = &gammaTensor;
  }
  if (beta != nullptr) {
    auto betaInfo = getArmTensorInfo(*beta, Arm_DataLayout::UNKNOWN);
    betaTensor.allocator()->init(betaInfo);
    betaPtr = &betaTensor;
  }

  // Configure batch normalization
  arm_compute::NEBatchNormalizationLayer batchnorm;
  batchnorm.configure(&inTensor, &outTensor, &meanTensor, &varTensor, betaPtr, gammaPtr, epsilon);

  // Import or allocate memory for input
  if (!input->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input, inTensor);
  }

  // Import or allocate memory for output
  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  // Import mean and variance
  if (!mean->hasPaddedBuffer() && !meanTensor.info()->has_padding()) {
    meanTensor.allocator()->import_memory(mean->buffer());
  } else {
    meanTensor.allocator()->allocate();
    copyToTensor(*mean, meanTensor);
  }

  if (!variance->hasPaddedBuffer() && !varTensor.info()->has_padding()) {
    varTensor.allocator()->import_memory(variance->buffer());
  } else {
    varTensor.allocator()->allocate();
    copyToTensor(*variance, varTensor);
  }

  // Import gamma if present
  if (gamma != nullptr) {
    if (!gamma->hasPaddedBuffer() && !gammaTensor.info()->has_padding()) {
      gammaTensor.allocator()->import_memory(gamma->buffer());
    } else {
      gammaTensor.allocator()->allocate();
      copyToTensor(*gamma, gammaTensor);
    }
  }

  // Import beta if present
  if (beta != nullptr) {
    if (!beta->hasPaddedBuffer() && !betaTensor.info()->has_padding()) {
      betaTensor.allocator()->import_memory(beta->buffer());
    } else {
      betaTensor.allocator()->allocate();
      copyToTensor(*beta, betaTensor);
    }
  }

  // Run batch normalization
  batchnorm.run();

  // Copy output if needed
  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(batchnorm, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto mean = INPUT_VARIABLE(1);
  auto variance = INPUT_VARIABLE(2);
  auto output = OUTPUT_VARIABLE(0);

  const bool applyScale = (bool)INT_ARG(0);
  const bool applyOffset = (bool)INT_ARG(1);

  NDArray* gamma = nullptr;
  NDArray* beta = nullptr;

  if (applyScale) gamma = INPUT_VARIABLE(3);
  if (applyOffset) beta = INPUT_VARIABLE(3 + (int)applyScale);

  const int numOfIntArgs = block.getIArguments()->size();
  std::vector<sd::LongType> axes;
  if (numOfIntArgs > 2)
    for (int i = 2; i < numOfIntArgs; ++i) axes.push_back(INT_ARG(i));
  else
    axes.push_back(input->rankOf() - 1);

  const int inRank = input->rankOf();

  Requirements req("ARMCOMPUTE BATCHNORM OP");
  req.expectEq(makeInfoVariable(axes.size(), "axes.size()"), 1) &&
      req.expectEq(makeInfoVariable(inRank, RANK_MSG_INPUT0), 4) &&
      req.expectIn(makeInfoVariable(axes[0], "axes#0"), {1, inRank - 1}) &&
      req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(mean->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(variance->dataType(), "input2#type"), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(input->stridesOf()[input->rankOf() - 1], "input0#lastStride"), 1) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->stridesOf()[output->rankOf() - 1], "output#lastStride"), 1);

  if (gamma != nullptr) {
    req.expectEq(makeInfoVariable(gamma->dataType(), "gamma#type"), DataType::FLOAT32);
  }
  if (beta != nullptr) {
    req.expectEq(makeInfoVariable(beta->dataType(), "beta#type"), DataType::FLOAT32);
  }

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
