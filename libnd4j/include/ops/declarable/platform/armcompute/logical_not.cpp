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
PLATFORM_IMPL(boolean_not, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Arm_TensorShape xShape, outShape;
  xShape.set_num_dimensions(x->rankOf());
  outShape.set_num_dimensions(output->rankOf());

  for (int i = 0; i < x->rankOf(); i++) {
    xShape[x->rankOf() - 1 - i] = x->sizeAt(i);
  }
  for (int i = 0; i < output->rankOf(); i++) {
    outShape[output->rankOf() - 1 - i] = output->sizeAt(i);
  }

  Arm_TensorInfo xInfo(xShape, 1, arm_compute::DataType::U8);
  Arm_TensorInfo outInfo(outShape, 1, arm_compute::DataType::U8);

  Arm_Tensor xTensor, outTensor;
  xTensor.allocator()->init(xInfo);
  outTensor.allocator()->init(outInfo);

  arm_compute::NELogicalNot logicalNot;
  logicalNot.configure(&xTensor, &outTensor);

  xTensor.allocator()->allocate();
  outTensor.allocator()->allocate();

  auto xPtr = reinterpret_cast<uint8_t*>(xTensor.buffer());
  for (sd::LongType i = 0; i < x->lengthOf(); i++) {
    xPtr[i] = x->e<bool>(i) ? 1 : 0;
  }

  logicalNot.run();

  auto outPtr = reinterpret_cast<uint8_t*>(outTensor.buffer());
  for (sd::LongType i = 0; i < output->lengthOf(); i++) {
    output->p(i, outPtr[i] != 0);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(boolean_not, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE BOOLEAN_NOT OP");
  req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::BOOL) &&
      req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->ordering(), ORDERING_MSG_INPUT), 'c');
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
