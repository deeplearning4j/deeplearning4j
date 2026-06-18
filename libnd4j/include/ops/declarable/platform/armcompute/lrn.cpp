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
#include <ops/declarable/helpers/convolutions.h>
#include <system/platform_boilerplate.h>

#include "armcomputeUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(lrn, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->rankOf() == 4, 0, "LRN ARMCOMPUTE: Input rank of 4 expected, but got %i instead", input->rankOf());

  double alpha = T_ARG(1);
  double beta = T_ARG(2);
  double bias = T_ARG(0);
  int depth = INT_ARG(0);

  // ARM Compute uses NCHW format for LRN
  auto dataLayout = arm_compute::DataLayout::NCHW;

  // Create tensor info
  auto inInfo = getArmTensorInfo(*input, dataLayout);
  auto outInfo = getArmTensorInfo(*output, dataLayout);

  Arm_Tensor inTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  outTensor.allocator()->init(outInfo);

  // ARM Compute LRN normalization info
  // norm_size is the full size (2 * depth + 1)
  arm_compute::NormalizationLayerInfo normInfo(
      arm_compute::NormType::CROSS_MAP,  // LRN across channels
      2 * depth + 1,                      // normalization size
      static_cast<float>(alpha * (2 * depth + 1)),  // alpha
      static_cast<float>(beta),                      // beta
      static_cast<float>(bias)                       // kappa (bias)
  );

  // Configure LRN
  arm_compute::NENormalizationLayer lrn;
  lrn.configure(&inTensor, &outTensor, normInfo);

  // Import or allocate memory for input
  if (!input->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input, inTensor);
  }

  // Import or allocate memory for output
  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  // Run LRN
  lrn.run();

  // Copy output if needed
  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(lrn, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE LRN OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 4) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(input->stridesOf()[input->rankOf() - 1], "input#lastStride"), 1) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->stridesOf()[output->rankOf() - 1], "output#lastStride"), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
