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
PLATFORM_IMPL(dequantize, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Get scale and zero point
  float scale = 1.0f;
  int32_t zeroPoint = 0;
  if (block.numT() > 0) scale = T_ARG(0);
  if (block.numI() > 0) zeroPoint = INT_ARG(0);

  // Input is quantized
  Arm_TensorShape inShape;
  inShape.set_num_dimensions(input->rankOf());
  for (int i = 0; i < input->rankOf(); i++) {
    inShape[input->rankOf() - 1 - i] = input->sizeAt(i);
  }

  arm_compute::QuantizationInfo quantInfo(scale, zeroPoint);
  Arm_TensorInfo inInfo(inShape, 1, arm_compute::DataType::QASYMM8, quantInfo);

  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);

  Arm_Tensor inTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  outTensor.allocator()->init(outInfo);

  arm_compute::NEDequantizationLayer dequantize;
  dequantize.configure(&inTensor, &outTensor);

  // Copy quantized input
  inTensor.allocator()->allocate();
  auto inPtr = reinterpret_cast<uint8_t*>(inTensor.buffer());
  for (sd::LongType i = 0; i < input->lengthOf(); i++) {
    inPtr[i] = static_cast<uint8_t>(input->e<int>(i));
  }

  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  dequantize.run();

  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(dequantize, ENGINE_CPU) {
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE DEQUANTIZE OP");
  req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), 0) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->stridesOf()[output->rankOf() - 1], "output#lastStride"), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
