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
PLATFORM_IMPL(instance_normalization, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  float epsilon = 1e-5f;
  if (block.numT() > 0) {
    epsilon = T_ARG(0);
  }

  bool useScale = false, useBias = false;
  NDArray* gamma = nullptr;
  NDArray* beta = nullptr;

  if (block.width() > 1) {
    gamma = INPUT_VARIABLE(1);
    useScale = true;
  }
  if (block.width() > 2) {
    beta = INPUT_VARIABLE(2);
    useBias = true;
  }

  auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::NCHW);
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::NCHW);

  Arm_Tensor inTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  outTensor.allocator()->init(outInfo);

  arm_compute::NEInstanceNormalizationLayer instanceNorm;

  if (useScale && useBias) {
    auto gammaInfo = getArmTensorInfo(*gamma, Arm_DataLayout::UNKNOWN);
    auto betaInfo = getArmTensorInfo(*beta, Arm_DataLayout::UNKNOWN);

    Arm_Tensor gammaTensor, betaTensor;
    gammaTensor.allocator()->init(gammaInfo);
    betaTensor.allocator()->init(betaInfo);

    instanceNorm.configure(&inTensor, &outTensor, epsilon);

    if (!gamma->hasPaddedBuffer() && !gammaTensor.info()->has_padding()) {
      gammaTensor.allocator()->import_memory(gamma->buffer());
    } else {
      gammaTensor.allocator()->allocate();
      copyToTensor(*gamma, gammaTensor);
    }

    if (!beta->hasPaddedBuffer() && !betaTensor.info()->has_padding()) {
      betaTensor.allocator()->import_memory(beta->buffer());
    } else {
      betaTensor.allocator()->allocate();
      copyToTensor(*beta, betaTensor);
    }
  } else {
    instanceNorm.configure(&inTensor, &outTensor, epsilon);
  }

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

  instanceNorm.run();

  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(instance_normalization, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE INSTANCE_NORMALIZATION OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 4) &&
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
