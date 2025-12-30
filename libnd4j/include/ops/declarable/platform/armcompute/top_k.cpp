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
PLATFORM_IMPL(top_k, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto values = OUTPUT_VARIABLE(0);
  auto indices = OUTPUT_VARIABLE(1);

  int k = INT_ARG(0);
  bool sorted = block.numB() > 0 ? B_ARG(0) : true;

  auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);
  auto valuesInfo = getArmTensorInfo(*values, Arm_DataLayout::UNKNOWN);

  // Indices are S32 in ARM Compute
  Arm_TensorShape indicesShape;
  indicesShape.set_num_dimensions(indices->rankOf());
  for (int i = 0; i < indices->rankOf(); i++) {
    indicesShape[indices->rankOf() - 1 - i] = indices->sizeAt(i);
  }
  Arm_TensorInfo indicesInfo(indicesShape, 1, arm_compute::DataType::S32);

  Arm_Tensor inTensor, valuesTensor, indicesTensor;
  inTensor.allocator()->init(inInfo);
  valuesTensor.allocator()->init(valuesInfo);
  indicesTensor.allocator()->init(indicesInfo);

  // Configure top-k - finds k largest values
  // Note: ARM Compute doesn't have direct TopK, this is a simplified version
  // using reduction operations. For full TopK, custom implementation needed.

  if (!input->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input, inTensor);
  }

  bool copyValues = false;
  if (!values->hasPaddedBuffer() && !valuesTensor.info()->has_padding()) {
    valuesTensor.allocator()->import_memory(values->buffer());
  } else {
    valuesTensor.allocator()->allocate();
    copyValues = true;
  }

  indicesTensor.allocator()->allocate();

  // For now, use a simple approach - this is a placeholder
  // Full TopK would require sorting which ARM Compute doesn't directly support
  // Falling back to CPU implementation in this case
  return sd::Status::KERNEL_FAILURE;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(top_k, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);

  // TopK is complex and may not be fully supported
  // Return false to use CPU fallback
  Requirements req("ARMCOMPUTE TOP_K OP");
  req.expectFalse(makeInfoVariable(true, "TopK not fully supported"));
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
