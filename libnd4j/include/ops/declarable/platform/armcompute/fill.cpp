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
PLATFORM_IMPL(fill, ENGINE_CPU) {
  auto output = OUTPUT_VARIABLE(0);

  // Get fill value
  float fillValue = 0.0f;
  if (block.width() > 1) {
    auto valueArr = INPUT_VARIABLE(1);
    fillValue = valueArr->e<float>(0);
  } else if (block.numT() > 0) {
    fillValue = T_ARG(0);
  }

  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);
  Arm_Tensor outTensor;
  outTensor.allocator()->init(outInfo);

  arm_compute::NEFill fill;
  fill.configure(&outTensor, arm_compute::PixelValue(fillValue));

  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  fill.run();

  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(fill, ENGINE_CPU) {
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE FILL OP");
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
