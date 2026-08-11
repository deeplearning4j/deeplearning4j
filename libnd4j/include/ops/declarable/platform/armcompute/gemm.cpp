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
// GEMM: C = alpha * A * B + beta * C
PLATFORM_IMPL(gemm, ENGINE_CPU) {
  auto A = INPUT_VARIABLE(0);
  auto B = INPUT_VARIABLE(1);
  auto C = OUTPUT_VARIABLE(0);

  NDArray* biasC = nullptr;
  if (block.width() > 2) {
    biasC = INPUT_VARIABLE(2);
  }

  float alpha = 1.0f, beta = 0.0f;
  bool transA = false, transB = false;

  if (block.numT() > 0) alpha = T_ARG(0);
  if (block.numT() > 1) beta = T_ARG(1);
  if (block.numI() > 0) transA = INT_ARG(0) != 0;
  if (block.numI() > 1) transB = INT_ARG(1) != 0;

  auto aInfo = getArmTensorInfo(*A, Arm_DataLayout::UNKNOWN);
  auto bInfo = getArmTensorInfo(*B, Arm_DataLayout::UNKNOWN);
  auto cInfo = getArmTensorInfo(*C, Arm_DataLayout::UNKNOWN);

  Arm_Tensor aTensor, bTensor, cTensor;
  aTensor.allocator()->init(aInfo);
  bTensor.allocator()->init(bInfo);
  cTensor.allocator()->init(cInfo);

  arm_compute::NEGEMM gemm;
  // GEMMInfo defaults both reshape flags to false.  The setter methods were
  // removed from newer ARM Compute releases, so keep this API-version-neutral.
  arm_compute::GEMMInfo gemmInfo;

  gemm.configure(&aTensor, &bTensor, nullptr, &cTensor, alpha, beta, gemmInfo);

  if (!A->hasPaddedBuffer() && !aTensor.info()->has_padding()) {
    aTensor.allocator()->import_memory(A->buffer());
  } else {
    aTensor.allocator()->allocate();
    copyToTensor(*A, aTensor);
  }

  if (!B->hasPaddedBuffer() && !bTensor.info()->has_padding()) {
    bTensor.allocator()->import_memory(B->buffer());
  } else {
    bTensor.allocator()->allocate();
    copyToTensor(*B, bTensor);
  }

  bool copyOutput = false;
  if (!C->hasPaddedBuffer() && !cTensor.info()->has_padding()) {
    cTensor.allocator()->import_memory(C->buffer());
  } else {
    cTensor.allocator()->allocate();
    copyOutput = true;
  }

  gemm.run();

  if (copyOutput) {
    copyFromTensor(cTensor, *C);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(gemm, ENGINE_CPU) {
  auto A = INPUT_VARIABLE(0);
  auto B = INPUT_VARIABLE(1);
  auto C = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE GEMM OP");
  req.expectEq(makeInfoVariable(A->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(B->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(C->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(A->rankOf(), RANK_MSG_INPUT), 2) &&
      req.expectEq(makeInfoVariable(B->rankOf(), RANK_MSG_INPUT), 2) &&
      req.expectEq(makeInfoVariable(A->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(B->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(C->ordering(), ORDERING_MSG_OUTPUT), 'c');
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
