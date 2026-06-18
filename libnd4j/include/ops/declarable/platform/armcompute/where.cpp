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
// Select operation: output = condition ? x : y
PLATFORM_IMPL(select, ENGINE_CPU) {
  auto condition = INPUT_VARIABLE(0);
  auto x = INPUT_VARIABLE(1);
  auto y = INPUT_VARIABLE(2);
  auto output = OUTPUT_VARIABLE(0);

  // Create condition tensor as U8
  Arm_TensorShape condShape;
  condShape.set_num_dimensions(condition->rankOf());
  for (int i = 0; i < condition->rankOf(); i++) {
    condShape[condition->rankOf() - 1 - i] = condition->sizeAt(i);
  }
  Arm_TensorInfo condInfo(condShape, 1, arm_compute::DataType::U8);

  auto xInfo = getArmTensorInfo(*x, Arm_DataLayout::UNKNOWN);
  auto yInfo = getArmTensorInfo(*y, Arm_DataLayout::UNKNOWN);
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);

  Arm_Tensor condTensor, xTensor, yTensor, outTensor;
  condTensor.allocator()->init(condInfo);
  xTensor.allocator()->init(xInfo);
  yTensor.allocator()->init(yInfo);
  outTensor.allocator()->init(outInfo);

  arm_compute::NESelect select;
  select.configure(&condTensor, &xTensor, &yTensor, &outTensor);

  // Allocate and copy condition
  condTensor.allocator()->allocate();
  auto condPtr = reinterpret_cast<uint8_t*>(condTensor.buffer());
  for (sd::LongType i = 0; i < condition->lengthOf(); i++) {
    condPtr[i] = condition->e<bool>(i) ? 255 : 0;
  }

  // Import or allocate x
  if (!x->hasPaddedBuffer() && !xTensor.info()->has_padding()) {
    xTensor.allocator()->import_memory(x->buffer());
  } else {
    xTensor.allocator()->allocate();
    copyToTensor(*x, xTensor);
  }

  // Import or allocate y
  if (!y->hasPaddedBuffer() && !yTensor.info()->has_padding()) {
    yTensor.allocator()->import_memory(y->buffer());
  } else {
    yTensor.allocator()->allocate();
    copyToTensor(*y, yTensor);
  }

  // Import or allocate output
  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  select.run();

  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(select, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(1);
  auto y = INPUT_VARIABLE(2);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE SELECT OP");
  req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(y->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(y->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c');
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
