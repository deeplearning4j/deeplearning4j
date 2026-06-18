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
PLATFORM_IMPL(layer_norm, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto gain = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  NDArray* bias = nullptr;
  if (block.width() > 2) {
    bias = INPUT_VARIABLE(2);
  }

  float epsilon = 1e-5f;
  if (block.numT() > 0) {
    epsilon = T_ARG(0);
  }

  // Get normalization axis
  std::vector<int> axes;
  if (block.numI() > 0) {
    for (int i = 0; i < block.numI(); i++) {
      int axis = INT_ARG(i);
      if (axis < 0) axis += input->rankOf();
      axes.push_back(axis);
    }
  } else {
    // Default: normalize over last axis
    axes.push_back(input->rankOf() - 1);
  }

  // Compute mean
  auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);
  Arm_Tensor inTensor;
  inTensor.allocator()->init(inInfo);

  if (!input->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input, inTensor);
  }

  // Convert axes to ARM Compute coordinates
  arm_compute::Coordinates reduction_axis;
  for (int axis : axes) {
    reduction_axis.set(reduction_axis.num_dimensions(), input->rankOf() - 1 - axis);
  }

  // Step 1: Compute mean
  Arm_Tensor meanTensor;
  auto meanInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);
  meanTensor.allocator()->init(meanInfo);
  meanTensor.allocator()->allocate();

  arm_compute::NEReduceMean reduceMean;
  reduceMean.configure(&inTensor, reduction_axis, true, &meanTensor);
  reduceMean.run();

  // Step 2: Subtract mean (x - mean)
  Arm_Tensor centeredTensor;
  centeredTensor.allocator()->init(inInfo);
  centeredTensor.allocator()->allocate();

  arm_compute::NEArithmeticSubtraction subtract;
  subtract.configure(&inTensor, &meanTensor, &centeredTensor, arm_compute::ConvertPolicy::SATURATE);
  subtract.run();

  // Step 3: Square centered values
  Arm_Tensor squaredTensor;
  squaredTensor.allocator()->init(inInfo);
  squaredTensor.allocator()->allocate();

  arm_compute::NEPixelWiseMultiplication square;
  square.configure(&centeredTensor, &centeredTensor, &squaredTensor, 1.0f,
                   arm_compute::ConvertPolicy::SATURATE, arm_compute::RoundingPolicy::TO_NEAREST_EVEN);
  square.run();

  // Step 4: Compute variance (mean of squared)
  Arm_Tensor varianceTensor;
  varianceTensor.allocator()->init(meanInfo);
  varianceTensor.allocator()->allocate();

  arm_compute::NEReduceMean reduceVariance;
  reduceVariance.configure(&squaredTensor, reduction_axis, true, &varianceTensor);
  reduceVariance.run();

  // Step 5: Add epsilon and compute rsqrt
  // variance + epsilon
  Arm_Tensor epsilonTensor;
  epsilonTensor.allocator()->init(meanInfo);
  epsilonTensor.allocator()->allocate();
  auto epsPtr = reinterpret_cast<float*>(epsilonTensor.buffer());
  for (size_t i = 0; i < output->lengthOf(); i++) {
    epsPtr[i] = epsilon;
  }

  Arm_Tensor varPlusEpsTensor;
  varPlusEpsTensor.allocator()->init(meanInfo);
  varPlusEpsTensor.allocator()->allocate();

  arm_compute::NEArithmeticAddition addEps;
  addEps.configure(&varianceTensor, &epsilonTensor, &varPlusEpsTensor, arm_compute::ConvertPolicy::SATURATE);
  addEps.run();

  // rsqrt(variance + epsilon)
  Arm_Tensor rsqrtTensor;
  rsqrtTensor.allocator()->init(meanInfo);
  rsqrtTensor.allocator()->allocate();

  arm_compute::NERsqrtLayer rsqrt;
  rsqrt.configure(&varPlusEpsTensor, &rsqrtTensor);
  rsqrt.run();

  // Step 6: Normalize: (x - mean) * rsqrt(var + eps)
  Arm_Tensor normalizedTensor;
  normalizedTensor.allocator()->init(inInfo);
  normalizedTensor.allocator()->allocate();

  arm_compute::NEPixelWiseMultiplication normalize;
  normalize.configure(&centeredTensor, &rsqrtTensor, &normalizedTensor, 1.0f,
                      arm_compute::ConvertPolicy::SATURATE, arm_compute::RoundingPolicy::TO_NEAREST_EVEN);
  normalize.run();

  // Step 7: Apply gain (scale)
  auto gainInfo = getArmTensorInfo(*gain, Arm_DataLayout::UNKNOWN);
  Arm_Tensor gainTensor;
  gainTensor.allocator()->init(gainInfo);
  if (!gain->hasPaddedBuffer() && !gainTensor.info()->has_padding()) {
    gainTensor.allocator()->import_memory(gain->buffer());
  } else {
    gainTensor.allocator()->allocate();
    copyToTensor(*gain, gainTensor);
  }

  Arm_Tensor scaledTensor;
  scaledTensor.allocator()->init(inInfo);

  bool copyOutput = false;
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);
  Arm_Tensor outTensor;
  outTensor.allocator()->init(outInfo);

  if (bias) {
    scaledTensor.allocator()->allocate();
    arm_compute::NEPixelWiseMultiplication scale;
    scale.configure(&normalizedTensor, &gainTensor, &scaledTensor, 1.0f,
                    arm_compute::ConvertPolicy::SATURATE, arm_compute::RoundingPolicy::TO_NEAREST_EVEN);
    scale.run();

    // Step 8: Add bias
    auto biasInfo = getArmTensorInfo(*bias, Arm_DataLayout::UNKNOWN);
    Arm_Tensor biasTensor;
    biasTensor.allocator()->init(biasInfo);
    if (!bias->hasPaddedBuffer() && !biasTensor.info()->has_padding()) {
      biasTensor.allocator()->import_memory(bias->buffer());
    } else {
      biasTensor.allocator()->allocate();
      copyToTensor(*bias, biasTensor);
    }

    if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
      outTensor.allocator()->import_memory(output->buffer());
    } else {
      outTensor.allocator()->allocate();
      copyOutput = true;
    }

    arm_compute::NEArithmeticAddition addBias;
    addBias.configure(&scaledTensor, &biasTensor, &outTensor, arm_compute::ConvertPolicy::SATURATE);
    addBias.run();
  } else {
    if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
      outTensor.allocator()->import_memory(output->buffer());
    } else {
      outTensor.allocator()->allocate();
      copyOutput = true;
    }

    arm_compute::NEPixelWiseMultiplication scale;
    scale.configure(&normalizedTensor, &gainTensor, &outTensor, 1.0f,
                    arm_compute::ConvertPolicy::SATURATE, arm_compute::RoundingPolicy::TO_NEAREST_EVEN);
    scale.run();
  }

  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(layer_norm, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE LAYER_NORM OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 0) &&
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
