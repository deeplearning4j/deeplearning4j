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

#include <numeric>

#include "armcomputeUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(matmul, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

  sd::LongType iSize = (sd::LongType)block.getIArguments()->size();
  int transX = iSize > 0 ? INT_ARG(0) : 0;
  int transY = iSize > 1 ? INT_ARG(1) : 0;
  const int transZ = iSize > 2 ? INT_ARG(2) : 0;

  // optional alpha and beta
  iSize = (sd::LongType)block.getTArguments()->size();
  float alpha = iSize > 0 ? static_cast<float>(T_ARG(0)) : 1.0f;
  float beta = iSize > 1 ? static_cast<float>(T_ARG(1)) : 0.0f;

  const sd::LongType xRank = x->rankOf();
  const sd::LongType yRank = y->rankOf();

  if (transZ) {
    x = INPUT_VARIABLE(1);
    y = INPUT_VARIABLE(0);
    bool temp = transX;
    transX = !transY;
    transY = !temp;
  }

  // ARM Compute GEMM expects 2D matrices
  REQUIRE_TRUE(xRank == 2 && yRank == 2, 0,
               "MATMUL ARMCOMPUTE OP: ARM Compute currently supports only 2D matrix multiplication, but got x rank = %i, y rank = %i !",
               xRank, yRank);

  // Create ARM Compute tensor info
  auto xInfo = getArmTensorInfo(*x, Arm_DataLayout::UNKNOWN);
  auto yInfo = getArmTensorInfo(*y, Arm_DataLayout::UNKNOWN);
  auto zInfo = getArmTensorInfo(*z, Arm_DataLayout::UNKNOWN);

  Arm_Tensor xTensor, yTensor, zTensor;
  xTensor.allocator()->init(xInfo);
  yTensor.allocator()->init(yInfo);
  zTensor.allocator()->init(zInfo);

  // Configure GEMM
  arm_compute::NEGEMM gemm;
  gemm.configure(&xTensor, &yTensor, nullptr, &zTensor, alpha, beta);

  // Import or allocate memory
  if (!x->hasPaddedBuffer() && !xTensor.info()->has_padding()) {
    xTensor.allocator()->import_memory(x->buffer());
  } else {
    xTensor.allocator()->allocate();
    copyToTensor(*x, xTensor);
  }

  if (!y->hasPaddedBuffer() && !yTensor.info()->has_padding()) {
    yTensor.allocator()->import_memory(y->buffer());
  } else {
    yTensor.allocator()->allocate();
    copyToTensor(*y, yTensor);
  }

  bool copyOutput = false;
  if (!z->hasPaddedBuffer() && !zTensor.info()->has_padding()) {
    zTensor.allocator()->import_memory(z->buffer());
  } else {
    zTensor.allocator()->allocate();
    copyOutput = true;
  }

  // Run GEMM
  gemm.run();

  // Copy output if needed
  if (copyOutput) {
    copyFromTensor(zTensor, *z);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(matmul, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  const sd::LongType xRank = x->rankOf();
  const sd::LongType yRank = y->rankOf();

  Requirements req("ARMCOMPUTE MATMUL OP");
  req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(y->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(xRank, RANK_MSG_INPUT0), 2) &&
      req.expectEq(makeInfoVariable(yRank, RANK_MSG_INPUT1), 2) &&
      req.expectEq(makeInfoVariable(x->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(x->stridesOf()[x->rankOf() - 1], "input0#lastStride"), 1) &&
      req.expectEq(makeInfoVariable(y->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectEq(makeInfoVariable(y->stridesOf()[y->rankOf() - 1], "input1#lastStride"), 1) &&
      req.expectEq(makeInfoVariable(z->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(z->stridesOf()[z->rankOf() - 1], "output#lastStride"), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
