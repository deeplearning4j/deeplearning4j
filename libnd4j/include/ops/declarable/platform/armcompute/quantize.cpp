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
PLATFORM_IMPL(quantize, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);

  // Output is quantized (QASYMM8)
  Arm_TensorShape outShape;
  outShape.set_num_dimensions(output->rankOf());
  for (int i = 0; i < output->rankOf(); i++) {
    outShape[output->rankOf() - 1 - i] = output->sizeAt(i);
  }

  // Get scale and zero point
  float scale = 1.0f;
  int32_t zeroPoint = 0;
  if (block.numT() > 0) scale = T_ARG(0);
  if (block.numI() > 0) zeroPoint = INT_ARG(0);

  arm_compute::QuantizationInfo quantInfo(scale, zeroPoint);
  Arm_TensorInfo outInfo(outShape, 1, arm_compute::DataType::QASYMM8, quantInfo);

  Arm_Tensor inTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  outTensor.allocator()->init(outInfo);

  arm_compute::NEQuantizationLayer quantize;
  quantize.configure(&inTensor, &outTensor);

  if (!input->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input, inTensor);
  }

  outTensor.allocator()->allocate();

  quantize.run();

  // Copy quantized output
  auto outPtr = reinterpret_cast<uint8_t*>(outTensor.buffer());
  for (sd::LongType i = 0; i < output->lengthOf(); i++) {
    output->p(i, static_cast<int>(outPtr[i]));
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(quantize, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE QUANTIZE OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
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
