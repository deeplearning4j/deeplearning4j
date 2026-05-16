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
PLATFORM_IMPL(xw_plus_b, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);      // [M, K]
  auto w = INPUT_VARIABLE(1);      // [K, N] or [N, K] depending on transpose
  auto b = INPUT_VARIABLE(2);      // [N]
  auto z = OUTPUT_VARIABLE(0);     // [M, N]

  if (x->isEmpty() || w->isEmpty() || b->isEmpty()) return sd::Status::OK;

  const int xRank = x->rankOf();
  const int wRank = w->rankOf();
  const int zRank = z->rankOf();

  REQUIRE_TRUE(xRank == 2, 0, "XW_PLUS_B ARMCOMPUTE: Input x array should have rank equal 2, but got instead %i!", xRank);
  REQUIRE_TRUE(wRank == 2, 0, "XW_PLUS_B ARMCOMPUTE: Input weights array should have rank equal 2, but got instead %i!", wRank);
  REQUIRE_TRUE(zRank == 2, 0, "XW_PLUS_B ARMCOMPUTE: Output array should have rank equal 2, but got instead %i!", zRank);
  REQUIRE_TRUE(1 == b->rankOf() && b->lengthOf() == z->sizeAt(-1), 0,
               "XW_PLUS_B ARMCOMPUTE: Input bias vector should be 1D and have proper dimension 1x%i. "
               "But got rank %i, and got length %i instead %i.",
               z->sizeAt(-1), b->rankOf(), b->lengthOf(), z->sizeAt(-1));

  // INT_ARG(1) is bTranspose (weight transpose). When bTranspose=0, user weights
  // are [K,N] and need transposing. When bTranspose=1, already in [N,K] form.
  const bool bShouldTransp = block.getIArguments()->size() > 1 ? (0 == INT_ARG(1)) : true;

  // Create ARM Compute tensor info
  auto xInfo = getArmTensorInfo(*x, Arm_DataLayout::UNKNOWN);
  auto wInfo = getArmTensorInfo(*w, Arm_DataLayout::UNKNOWN);
  auto bInfo = getArmTensorInfo(*b, Arm_DataLayout::UNKNOWN);
  auto zInfo = getArmTensorInfo(*z, Arm_DataLayout::UNKNOWN);

  Arm_Tensor xTensor, wTensor, bTensor, zTensor;
  xTensor.allocator()->init(xInfo);
  wTensor.allocator()->init(wInfo);
  bTensor.allocator()->init(bInfo);
  zTensor.allocator()->init(zInfo);

  // Configure fully connected layer
  arm_compute::NEFullyConnectedLayer fc;
  arm_compute::FullyConnectedLayerInfo fcInfo;
  fcInfo.transpose_weights = bShouldTransp;

  fc.configure(&xTensor, &wTensor, &bTensor, &zTensor, fcInfo);

  // Import or allocate memory for input x
  if (!x->hasPaddedBuffer() && !xTensor.info()->has_padding()) {
    xTensor.allocator()->import_memory(x->buffer());
  } else {
    xTensor.allocator()->allocate();
    copyToTensor(*x, xTensor);
  }

  // Import or allocate memory for weights
  if (!w->hasPaddedBuffer() && !wTensor.info()->has_padding()) {
    wTensor.allocator()->import_memory(w->buffer());
  } else {
    wTensor.allocator()->allocate();
    copyToTensor(*w, wTensor);
  }

  // Import or allocate memory for bias
  if (!b->hasPaddedBuffer() && !bTensor.info()->has_padding()) {
    bTensor.allocator()->import_memory(b->buffer());
  } else {
    bTensor.allocator()->allocate();
    copyToTensor(*b, bTensor);
  }

  // Import or allocate memory for output
  bool copyOutput = false;
  if (!z->hasPaddedBuffer() && !zTensor.info()->has_padding()) {
    zTensor.allocator()->import_memory(z->buffer());
  } else {
    zTensor.allocator()->allocate();
    copyOutput = true;
  }

  // Run fully connected layer
  fc.run();

  // Copy output if needed
  if (copyOutput) {
    copyFromTensor(zTensor, *z);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(xw_plus_b, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto w = INPUT_VARIABLE(1);
  auto b = INPUT_VARIABLE(2);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE XW_PLUS_B OP");
  req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(w->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(b->dataType(), "bias#type"), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 2) &&
      req.expectEq(makeInfoVariable(w->rankOf(), RANK_MSG_INPUT1), 2) &&
      req.expectEq(makeInfoVariable(z->rankOf(), RANK_MSG_OUTPUT), 2) &&
      req.expectEq(makeInfoVariable(x->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(x->stridesOf()[x->rankOf() - 1], "input0#lastStride"), 1) &&
      req.expectEq(makeInfoVariable(w->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectEq(makeInfoVariable(w->stridesOf()[w->rankOf() - 1], "input1#lastStride"), 1) &&
      req.expectEq(makeInfoVariable(z->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(z->stridesOf()[z->rankOf() - 1], "output#lastStride"), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
