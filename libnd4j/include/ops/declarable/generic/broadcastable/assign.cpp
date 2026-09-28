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
// Created by raver119 on 24.11.17.
//

#include <system/op_boilerplate.h>
#include <helpers/StringUtils.h>
#if NOT_EXCLUDED(OP_assign)

#include <ops/declarable/headers/broadcastable.h>
#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/helpers/assign.h>

namespace sd {
namespace ops {
BROADCASTABLE_OP_IMPL(assign, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = block.width() < 2 ? x: INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  // Check if any array is of string type
  if (x->isS() || y->isS() || z->isS()) {
    // Handle string broadcast at high level
    StringUtils::broadcastStringAssign(x,z);
    return Status::OK;
  }

  NDArray *castedX = x->dataType() == z->dataType() ? x : x->cast(z->dataType());
  NDArray *castedY = y->dataType() == z->dataType() ? y : y->cast(z->dataType());

  ArrayOptions::validateSingleDataType(ArrayOptions::dataType(castedX->shapeInfo()));
  ArrayOptions::validateSingleDataType(ArrayOptions::extra(castedY->shapeInfo()));
  ArrayOptions::validateSingleDataType(ArrayOptions::extra(z->shapeInfo()));

  // FLOAT8/FLOAT8_E5M2 are deliberately excluded from SD_COMMON_TYPES (see
  // helpers/cpu/assign.cpp, helpers/cuda/assign.cu) to avoid instantiating every
  // broadcastable op's PairwiseTransform/BroadcastHelper templates for both FP8
  // encodings. They are handled via the dedicated helpers::assign() dispatch
  // instead (already used by the "cast" op for the same reason). A straight,
  // non-broadcast copy - e.g. dup()/assign() onto a reordered ('f') view, which
  // has no PairwiseTransform_THRICE<FLOAT8,...> instantiation - must go through
  // that helper rather than BroadcastHelper::broadcastApply.
  if (castedX->isSameShape(z) &&
      (castedX->dataType() == DataType::FLOAT8 || castedX->dataType() == DataType::FLOAT8_E5M2 ||
       z->dataType() == DataType::FLOAT8 || z->dataType() == DataType::FLOAT8_E5M2)) {
    helpers::assign(block.launchContext(), z, castedX);
    if (castedX != x) delete castedX;
    if (castedY != y) delete castedY;
    return Status::OK;
  }

  auto tZ = BroadcastHelper::broadcastApply(BroadcastOpsTuple::Assign(), castedX, castedY, z);

  if (tZ != z) {
    OVERWRITE_RESULT(tZ);
  }

  // Cleanup casted arrays if they were allocated
  if (castedX != x) delete castedX;
  if (castedY != y) delete castedY;

  return Status::OK;
}
DECLARE_SYN(set, assign);
DECLARE_SYN(copy, assign);

DECLARE_TYPES(assign) {
  // ALL_FLOATS deliberately excludes the FP8 storage types (FLOAT8/FLOAT8_E5M2):
  // most float transforms (sigmoid, tanh, exp, ...) are not meaningful on them.
  // assign is a plain copy/type-pun though, and both the CPU and CUDA assign
  // helpers (helpers/cpu/assign.cpp, helpers/cuda/assign.cu) already implement
  // every FLOAT8/FLOAT8_E5M2 <-> FLOAT8/FLOAT8_E5M2 combination correctly - only
  // this op-descriptor type gate was rejecting them before the kernel ever ran,
  // which is what made dup('f')/view layout on FLOAT8 fail on CPU (and would
  // fail identically on CUDA if it every routed FLOAT8 dup through this op).
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS,ALL_FLOATS,ALL_STRINGS,BOOL,sd::DataType::FLOAT8,sd::DataType::FLOAT8_E5M2})
      ->setAllowedInputTypes(1, {ALL_INTS,ALL_FLOATS,ALL_STRINGS,BOOL,sd::DataType::FLOAT8,sd::DataType::FLOAT8_E5M2})
      ->setAllowedOutputTypes(0, {ALL_INTS,ALL_FLOATS,ALL_STRINGS,BOOL,sd::DataType::FLOAT8,sd::DataType::FLOAT8_E5M2});
  getOpDescriptor()->addTraits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}

DECLARE_TYPES(assign_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_INTS,ALL_FLOATS,ALL_STRINGS});
}

CUSTOM_OP_IMPL(assign_bp, 3, 2, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = block.width() < 2 ? x->dup(x->ordering(), false) : INPUT_VARIABLE(1);  // dup() already returns NDArray*
  auto epsNext = INPUT_VARIABLE(2);

  auto gradX = OUTPUT_VARIABLE(0);
  auto gradY = OUTPUT_VARIABLE(1);

  float zero = 0.0f;
  gradX->assign(zero);

  if (x->isSameShape(y)) {
    gradY->assign(epsNext);
  } else if (y->isScalar()) {
    auto sum = epsNext->reduceNumber(reduce::Sum);
    gradY->assign(sum);
    delete sum;
  } else {
    // broadcastable
    auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

    if (axisY.size() > 0) {
      auto sum = epsNext->reduceAlongDimension(reduce::Sum, &axisY);
      gradY->assign(sum);
      delete sum;
    } else
      gradY->assign(epsNext);
  }

  return Status::OK;
}

DECLARE_SHAPE_FN(assign_bp) {
  auto x = inputShape->at(0);
  auto y = inputShape->at(1);
  auto e = inputShape->at(2);

  // eps always has shape of x
  // grad always has shape of y
  auto shapeList = SHAPELIST(CONSTANT(x), CONSTANT(y));

  return shapeList;
}
}  // namespace ops
}  // namespace sd

#endif
