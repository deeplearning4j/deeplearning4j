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

#include <helpers/ConstantShapeHelper.h>
#include <ops/declarable/headers/broadcastable.h>
#include <ops/declarable/generic/helpers/BroadcastHelper.h>

namespace sd {
namespace ops {
#if defined(HAS_FLOAT8) && defined(HAS_UINT8)
static bool isFp8(DataType dtype) { return dtype == FLOAT8 || dtype == FLOAT8_E5M2; }

// A one-byte UINT8 view of the array's own storage: same buffer, offset, shape and strides.
static NDArray* storageBytes(NDArray* array) {
  return new NDArray(array->dataBuffer(), ConstantShapeHelper::getInstance().castToDataType(array->shapeInfo(), UINT8),
                     array->getContext(), array->offset());
}
#endif

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

#if defined(HAS_FLOAT8) && defined(HAS_UINT8)
  // FP8 has no arithmetic kernels in the pairwise/scalar/broadcast type lists, so an assign into an
  // FP8 output is a storage copy between arrays of that one FP8 dtype. It runs over UINT8 views of
  // the same storage, which copies every encoding (NaN, -0) bit for bit. An FP8 input with a
  // common-type output takes the cast below instead, whose TransformAny FP8 pairs convert it.
  if (isFp8(z->dataType())) {
    REQUIRE_TRUE(x->dataType() == z->dataType() && y->dataType() == z->dataType(), 0,
                 "ASSIGN OP: an FP8 output copies storage and needs its dtype for every operand (cast converts "
                 "into FP8), got x=%s, y=%s, z=%s",
                 DataTypeUtils::asString(x->dataType()).c_str(), DataTypeUtils::asString(y->dataType()).c_str(),
                 DataTypeUtils::asString(z->dataType()).c_str());
    if (x->isEmpty() || y->isEmpty()) return Status::OK;

    NDArray* xBytes = storageBytes(x);
    NDArray* yBytes = storageBytes(y);
    NDArray* zBytes = storageBytes(z);
    auto result = BroadcastHelper::broadcastApply(BroadcastOpsTuple::Assign(), xBytes, yBytes, zBytes);
    // Any other non-null result is a new byte array of y's shape (scalar x, larger y, z of another shape).
    const bool wroteOutput = result == zBytes;
    if (result != nullptr && !wroteOutput) delete result;
    delete xBytes;
    delete yBytes;
    delete zBytes;
    if (result == nullptr) return Status::KERNEL_FAILURE;
    REQUIRE_TRUE(wroteOutput, 0, "ASSIGN OP: FP8 assign needs an output of the broadcast shape %s, got %s",
                 ShapeUtils::shapeAsString(y).c_str(), ShapeUtils::shapeAsString(z).c_str());
    return Status::OK;
  }
#endif

  NDArray *castedX = x->dataType() == z->dataType() ? x : x->cast(z->dataType());
  NDArray *castedY = y->dataType() == z->dataType() ? y : y->cast(z->dataType());

  ArrayOptions::validateSingleDataType(ArrayOptions::dataType(castedX->shapeInfo()));
  ArrayOptions::validateSingleDataType(ArrayOptions::extra(castedY->shapeInfo()));
  ArrayOptions::validateSingleDataType(ArrayOptions::extra(z->shapeInfo()));

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
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS,ALL_FLOATS,ALL_STRINGS,BOOL})
      ->setAllowedInputTypes(1, {ALL_INTS,ALL_FLOATS,ALL_STRINGS,BOOL})
      ->setAllowedOutputTypes(0, {ALL_INTS,ALL_FLOATS,ALL_STRINGS,BOOL});
#if defined(HAS_FLOAT8) && defined(HAS_UINT8)
  // ALL_FLOATS excludes the FP8 storage types. An FP8 output is a same-dtype storage copy, which is
  // what dup('f') and view layout on FP8 need; an FP8 input converts into a common-type output.
  for (auto fp8 : {FLOAT8, FLOAT8_E5M2}) {
    getOpDescriptor()->setAllowedInputTypes(0, fp8)->setAllowedInputTypes(1, fp8)->setAllowedOutputTypes(0, fp8);
  }
#endif
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
