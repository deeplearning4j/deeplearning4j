/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by raver119 on 12.10.2017.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_maximum)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/helpers/minimax.h>
namespace sd {
namespace ops {
BROADCASTABLE_OP_IMPL(maximum, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  BROADCAST_CHECK_EMPTY(x, y, z);

  auto tZ = BroadcastHelper::broadcastApply(BROADCAST(MaxPairwise), x, y, z);
  if (tZ == nullptr)
    return Status::KERNEL_FAILURE;
  else if (tZ != z) {
    OVERWRITE_RESULT(tZ);
  }

  return Status::OK;
}

DECLARE_TYPES(maximum) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, ANY)
      ->setAllowedOutputTypes(0, INHERIT);
}

DECLARE_TYPES(maximum_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(maximum_bp, 3, 2, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto epsNext = INPUT_VARIABLE(2);

  auto gradX = OUTPUT_VARIABLE(0);
  auto gradY = OUTPUT_VARIABLE(1);

  helpers::maximumBPFunctor(block.launchContext(), x, y, epsNext, gradX, gradY);
  return Status::OK;
}

DECLARE_SHAPE_FN(maximum_bp) {
  auto x = inputShape->at(0);
  auto y = inputShape->at(1);
  auto e = inputShape->at(2);

  // eps always has shape of x
  // grad always has shape of y
  return SHAPELIST(CONSTANT(x), CONSTANT(y));
}
}  // namespace ops
}  // namespace sd

#endif
