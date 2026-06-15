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
PLATFORM_IMPL(multiply, ENGINE_CPU) {
  auto input0 = INPUT_VARIABLE(0);
  auto input1 = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  // Create tensor info
  auto in0Info = getArmTensorInfo(*input0, Arm_DataLayout::UNKNOWN);
  auto in1Info = getArmTensorInfo(*input1, Arm_DataLayout::UNKNOWN);
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);

  Arm_Tensor in0Tensor, in1Tensor, outTensor;
  in0Tensor.allocator()->init(in0Info);
  in1Tensor.allocator()->init(in1Info);
  outTensor.allocator()->init(outInfo);

  // Configure multiply (scale = 1.0, no rounding)
  arm_compute::NEPixelWiseMultiplication multiply;
  multiply.configure(&in0Tensor, &in1Tensor, &outTensor, 1.0f, arm_compute::ConvertPolicy::SATURATE,
                     arm_compute::RoundingPolicy::TO_ZERO);

  // Import or allocate memory for input0
  if (!input0->hasPaddedBuffer() && !in0Tensor.info()->has_padding()) {
    in0Tensor.allocator()->import_memory(input0->buffer());
  } else {
    in0Tensor.allocator()->allocate();
    copyToTensor(*input0, in0Tensor);
  }

  // Import or allocate memory for input1
  if (!input1->hasPaddedBuffer() && !in1Tensor.info()->has_padding()) {
    in1Tensor.allocator()->import_memory(input1->buffer());
  } else {
    in1Tensor.allocator()->allocate();
    copyToTensor(*input1, in1Tensor);
  }

  // Import or allocate memory for output
  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  // Run multiply
  multiply.run();

  // Copy output if needed
  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(multiply, ENGINE_CPU) {
  auto input0 = INPUT_VARIABLE(0);
  auto input1 = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  // Check shapes match (no broadcasting support in this simple implementation)
  bool shapesMatch = input0->isSameShape(input1) && input0->isSameShape(output);

  Requirements req("ARMCOMPUTE MULTIPLY OP");
  req.expectTrue(makeInfoVariable(shapesMatch, "shapes match"), NO_MSG) &&
      req.expectEq(makeInfoVariable(input0->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(input1->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(input0->rankOf(), RANK_MSG_INPUT0), arm_compute::MAX_DIMS) &&
      req.expectEq(makeInfoVariable(input0->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(input0->stridesOf()[input0->rankOf() - 1], "input0#lastStride"), 1) &&
      req.expectEq(makeInfoVariable(input1->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectEq(makeInfoVariable(input1->stridesOf()[input1->rankOf() - 1], "input1#lastStride"), 1) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->stridesOf()[output->rankOf() - 1], "output#lastStride"), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
