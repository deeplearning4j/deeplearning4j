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
PLATFORM_IMPL(crop_and_resize, ENGINE_CPU) {
  auto image = INPUT_VARIABLE(0);
  auto boxes = INPUT_VARIABLE(1);
  auto boxIndices = INPUT_VARIABLE(2);
  auto cropSize = INPUT_VARIABLE(3);
  auto output = OUTPUT_VARIABLE(0);

  // Get method (0 = bilinear, 1 = nearest)
  int method = 0;
  if (block.numI() > 0) {
    method = INT_ARG(0);
  }

  // Get extrapolation value
  float extrapolationValue = 0.0f;
  if (block.numT() > 0) {
    extrapolationValue = T_ARG(0);
  }

  auto imageInfo = getArmTensorInfo(*image, Arm_DataLayout::NHWC);
  auto boxesInfo = getArmTensorInfo(*boxes, Arm_DataLayout::UNKNOWN);
  auto boxIndicesInfo = getArmTensorInfo(*boxIndices, Arm_DataLayout::UNKNOWN);
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::NHWC);

  Arm_Tensor imageTensor, boxesTensor, boxIndicesTensor, outTensor;
  imageTensor.allocator()->init(imageInfo);
  boxesTensor.allocator()->init(boxesInfo);
  boxIndicesTensor.allocator()->init(boxIndicesInfo);
  outTensor.allocator()->init(outInfo);

  arm_compute::NECropResize cropResize;
  arm_compute::InterpolationPolicy interpolation = method == 0 ?
      arm_compute::InterpolationPolicy::BILINEAR :
      arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR;

  cropResize.configure(&imageTensor, &boxesTensor, &boxIndicesTensor, &outTensor,
                        interpolation, extrapolationValue);

  // Allocate tensors
  if (!image->hasPaddedBuffer() && !imageTensor.info()->has_padding()) {
    imageTensor.allocator()->import_memory(image->buffer());
  } else {
    imageTensor.allocator()->allocate();
    copyToTensor(*image, imageTensor);
  }

  boxesTensor.allocator()->allocate();
  copyToTensor(*boxes, boxesTensor);

  boxIndicesTensor.allocator()->allocate();
  auto idxPtr = reinterpret_cast<int32_t*>(boxIndicesTensor.buffer());
  for (sd::LongType i = 0; i < boxIndices->lengthOf(); i++) {
    idxPtr[i] = boxIndices->e<int32_t>(i);
  }

  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  cropResize.run();

  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(crop_and_resize, ENGINE_CPU) {
  auto image = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE CROP_AND_RESIZE OP");
  req.expectEq(makeInfoVariable(image->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(image->rankOf(), RANK_MSG_INPUT), 4) &&
      req.expectEq(makeInfoVariable(image->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(image->stridesOf()[image->rankOf() - 1], "input#lastStride"), 1) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->stridesOf()[output->rankOf() - 1], "output#lastStride"), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
