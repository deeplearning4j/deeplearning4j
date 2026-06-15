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
// ScaleAdd: output = a * input1 + b * input2
PLATFORM_IMPL(scaleadd, ENGINE_CPU) {
  auto input1 = INPUT_VARIABLE(0);
  auto input2 = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  float alpha = T_ARG(0);
  float beta = T_ARG(1);

  auto in1Info = getArmTensorInfo(*input1, Arm_DataLayout::UNKNOWN);
  auto in2Info = getArmTensorInfo(*input2, Arm_DataLayout::UNKNOWN);
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);

  Arm_Tensor in1Tensor, in2Tensor, outTensor;
  in1Tensor.allocator()->init(in1Info);
  in2Tensor.allocator()->init(in2Info);
  outTensor.allocator()->init(outInfo);

  // ARM Compute doesn't have direct scale_add, use ArithmeticAddition with accumulator
  // We'll compute it manually using the buffer

  if (!input1->hasPaddedBuffer() && !in1Tensor.info()->has_padding()) {
    in1Tensor.allocator()->import_memory(input1->buffer());
  } else {
    in1Tensor.allocator()->allocate();
    copyToTensor(*input1, in1Tensor);
  }

  if (!input2->hasPaddedBuffer() && !in2Tensor.info()->has_padding()) {
    in2Tensor.allocator()->import_memory(input2->buffer());
  } else {
    in2Tensor.allocator()->allocate();
    copyToTensor(*input2, in2Tensor);
  }

  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  // Compute: output = alpha * input1 + beta * input2
  auto in1Ptr = reinterpret_cast<float*>(in1Tensor.buffer());
  auto in2Ptr = reinterpret_cast<float*>(in2Tensor.buffer());
  auto outPtr = reinterpret_cast<float*>(outTensor.buffer());

  for (sd::LongType i = 0; i < input1->lengthOf(); i++) {
    outPtr[i] = alpha * in1Ptr[i] + beta * in2Ptr[i];
  }

  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(scaleadd, ENGINE_CPU) {
  auto input1 = INPUT_VARIABLE(0);
  auto input2 = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE SCALEADD OP");
  req.expectEq(makeInfoVariable(input1->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(input2->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(input1->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(input1->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(input1->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(input2->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectTrue(makeInfoVariable(input1->isSameShape(input2), "same shape"));
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
