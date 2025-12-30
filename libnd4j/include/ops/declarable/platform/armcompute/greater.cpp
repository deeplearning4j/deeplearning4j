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
PLATFORM_IMPL(greater, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  // Create tensor info
  auto xInfo = getArmTensorInfo(*x, Arm_DataLayout::UNKNOWN);
  auto yInfo = getArmTensorInfo(*y, Arm_DataLayout::UNKNOWN);

  // Output is boolean but ARM Compute uses U8 for comparison results
  Arm_TensorShape outShape;
  outShape.set_num_dimensions(output->rankOf());
  for (int i = 0; i < output->rankOf(); i++) {
    outShape[output->rankOf() - 1 - i] = output->sizeAt(i);
  }
  Arm_TensorInfo outInfo(outShape, 1, arm_compute::DataType::U8);

  Arm_Tensor xTensor, yTensor, outTensor;
  xTensor.allocator()->init(xInfo);
  yTensor.allocator()->init(yInfo);
  outTensor.allocator()->init(outInfo);

  // Configure comparison
  arm_compute::NEElementwiseComparison compare;
  compare.configure(&xTensor, &yTensor, &outTensor, arm_compute::ComparisonOperation::Greater);

  // Import or allocate memory for x
  if (!x->hasPaddedBuffer() && !xTensor.info()->has_padding()) {
    xTensor.allocator()->import_memory(x->buffer());
  } else {
    xTensor.allocator()->allocate();
    copyToTensor(*x, xTensor);
  }

  // Import or allocate memory for y
  if (!y->hasPaddedBuffer() && !yTensor.info()->has_padding()) {
    yTensor.allocator()->import_memory(y->buffer());
  } else {
    yTensor.allocator()->allocate();
    copyToTensor(*y, yTensor);
  }

  // Allocate output
  outTensor.allocator()->allocate();

  // Run comparison
  compare.run();

  // Copy results to output (convert U8 to bool)
  auto outPtr = reinterpret_cast<uint8_t*>(outTensor.buffer());
  for (sd::LongType i = 0; i < output->lengthOf(); i++) {
    output->p(i, outPtr[i] != 0);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(greater, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);

  Requirements req("ARMCOMPUTE GREATER OP");
  req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(y->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(x->stridesOf()[x->rankOf() - 1], "x#lastStride"), 1) &&
      req.expectEq(makeInfoVariable(y->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(y->stridesOf()[y->rankOf() - 1], "y#lastStride"), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
