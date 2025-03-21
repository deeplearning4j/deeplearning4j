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
//  @author raver119@gmail.com
//  modified by sgazeos@gmail.com with backprop implementation.
//
#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_floormod)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/BroadcastHelper.h>

namespace sd {
namespace ops {
BROADCASTABLE_OP_IMPL(floormod, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  BROADCAST_CHECK_EMPTY(x, y, z);

  REQUIRE_TRUE(!y->isB(), 0, "FLOORMOD OP: you can't divide by bool array!");
  auto tZ = BroadcastHelper::broadcastApply(BROADCAST(FloorMod), x, y, z);
  if (tZ == nullptr)
    return Status::KERNEL_FAILURE;
  else if (tZ != z) {
    OVERWRITE_RESULT(tZ);
  }

  return Status::OK;
}

DECLARE_TYPES(floormod) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, ANY)
      ->setAllowedOutputTypes(0, INHERIT);
}

DECLARE_TYPES(floormod_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(floormod_bp, 3, 2, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto epsNext = INPUT_VARIABLE(2);

  auto gradX = OUTPUT_VARIABLE(0);
  auto gradY = OUTPUT_VARIABLE(1);
  gradX->assign(*epsNext);

  NDArray temp(*epsNext);
  BroadcastHelper::broadcastApply(BROADCAST(FloorMod), x, y, &temp);
  if (gradY->rankOf() == gradX->rankOf()) {
    epsNext->applyPairwiseTransform(pairwise::Multiply, &temp, gradY);
  } else  { // epsNext is greater than gradY
    std::vector<LongType> dims(epsNext->rankOf() * 2);
    LongType gap = epsNext->rankOf() - gradY->rankOf();
    for (LongType d = 0; d < gap; d++) {
      dims[d * 2 + 1] = 1;
    }
    auto tempIn((temp)(dims));
    NDArray negTempIn = -tempIn;
    (*epsNext)(dims).applyPairwiseTransform(pairwise::Multiply, &negTempIn, gradY);
  }
  return Status::OK;
}

DECLARE_SHAPE_FN(floormod_bp) {
  auto x = inputShape->at(0);
  auto y = inputShape->at(1);
  auto e = inputShape->at(2);

  // eps always has shape of x
  // grad always has shape of y

  LongType* shapeE;
  LongType* shapeG;

  COPY_SHAPE(x, shapeE);
  COPY_SHAPE(y, shapeG);

  return SHAPELIST(CONSTANT(shapeE), CONSTANT(shapeG));
}
}  // namespace ops
}  // namespace sd

#endif
