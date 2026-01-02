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
// Bias Add - add bias to input along channel dimension
PLATFORM_IMPL(biasadd, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto bias = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  bool isNCHW = block.getIArguments()->size() > 0 ? INT_ARG(0) == 0 : true;

  // Use ARM Compute's arithmetic addition with broadcasting
  auto inInfo = getArmTensorInfo(*input, isNCHW ? Arm_DataLayout::NCHW : Arm_DataLayout::NHWC);
  auto outInfo = getArmTensorInfo(*output, isNCHW ? Arm_DataLayout::NCHW : Arm_DataLayout::NHWC);

  Arm_Tensor inTensor, biasTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  outTensor.allocator()->init(outInfo);

  // Create bias tensor info with proper broadcasting shape
  Arm_TensorShape biasShape;
  biasShape.set_num_dimensions(1);
  biasShape[0] = bias->lengthOf();
  Arm_TensorInfo biasInfo(biasShape, 1, arm_compute::DataType::F32);
  biasTensor.allocator()->init(biasInfo);

  arm_compute::NEArithmeticAddition biasAdd;
  biasAdd.configure(&inTensor, &biasTensor, &outTensor, arm_compute::ConvertPolicy::SATURATE);

  // Import/allocate input
  if (!input->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input, inTensor);
  }

  // Import/allocate bias
  if (!bias->hasPaddedBuffer() && !biasTensor.info()->has_padding()) {
    biasTensor.allocator()->import_memory(bias->buffer());
  } else {
    biasTensor.allocator()->allocate();
    auto bPtr = reinterpret_cast<float*>(biasTensor.buffer());
    for (sd::LongType i = 0; i < bias->lengthOf(); i++) {
      bPtr[i] = bias->e<float>(i);
    }
  }

  // Import/allocate output
  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  biasAdd.run();

  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(biasadd, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto bias = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE BIASADD OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(bias->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 1) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c');
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
