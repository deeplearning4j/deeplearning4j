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
PLATFORM_IMPL(pow, ENGINE_CPU) {
  auto base = INPUT_VARIABLE(0);
  auto exponent = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  // Create tensor info
  auto baseInfo = getArmTensorInfo(*base, Arm_DataLayout::UNKNOWN);
  auto expInfo = getArmTensorInfo(*exponent, Arm_DataLayout::UNKNOWN);
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);

  Arm_Tensor baseTensor, expTensor, outTensor;
  baseTensor.allocator()->init(baseInfo);
  expTensor.allocator()->init(expInfo);
  outTensor.allocator()->init(outInfo);

  // Configure power
  arm_compute::NEElementwisePower power;
  power.configure(&baseTensor, &expTensor, &outTensor);

  // Import or allocate memory for base
  if (!base->hasPaddedBuffer() && !baseTensor.info()->has_padding()) {
    baseTensor.allocator()->import_memory(base->buffer());
  } else {
    baseTensor.allocator()->allocate();
    copyToTensor(*base, baseTensor);
  }

  // Import or allocate memory for exponent
  if (!exponent->hasPaddedBuffer() && !expTensor.info()->has_padding()) {
    expTensor.allocator()->import_memory(exponent->buffer());
  } else {
    expTensor.allocator()->allocate();
    copyToTensor(*exponent, expTensor);
  }

  // Import or allocate memory for output
  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  // Run power
  power.run();

  // Copy output if needed
  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(pow, ENGINE_CPU) {
  auto base = INPUT_VARIABLE(0);
  auto exponent = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE POW OP");
  req.expectEq(makeInfoVariable(base->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(exponent->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(base->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(base->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(base->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(base->stridesOf()[base->rankOf() - 1], "base#lastStride"), 1) &&
      req.expectEq(makeInfoVariable(exponent->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(exponent->stridesOf()[exponent->rankOf() - 1], "exp#lastStride"), 1) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->stridesOf()[output->rankOf() - 1], "output#lastStride"), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
