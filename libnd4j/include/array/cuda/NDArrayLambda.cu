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

//
// @author raver119@gmail.com
//
#include <array/DataType.h>
#include <array/DataTypeUtils.h>
#include <array/NDArray.h>

#include <execution/ThreadPool.h>
#include <helpers/DebugHelper.h>
#include <loops/legacy_ops.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

#include "helpers/ShapeUtils.h"

namespace sd {

// ============================================================================
//  NDArray lambda operations on the CUDA backend.
//
//  A user-supplied std::function holds host code (and host-captured state), so
//  it CANNOT be invoked from a CUDA __global__ kernel -- there is no way to ship
//  an arbitrary std::function to the device. The previous implementation launched
//  placeholder kernels that silently copied x -> z (ignoring the function and the
//  second/third operands), so every CUDA lambda call returned wrong results with
//  no error. That silent landmine is removed here.
//
//  These methods now execute the function on the host (the only place the
//  std::function can run): inputs are synced to host via preparePrimaryUse, the
//  function is applied element-wise with a shared canonical (C-order) coordinate
//  decomposition -- identical pairing semantics to the CPU implementation, so
//  operands of differing ordering ('c' vs 'f') are never mis-paired -- and the
//  result is published back to the device via registerPrimaryUse.
//
//  Performance-critical paths (e.g. activation derivatives) must NOT rely on
//  these; they use real device ops. Lambdas are a correctness fallback for
//  arbitrary user functions where a host round-trip is acceptable.
// ============================================================================

template <typename T>
void NDArray::applyLambda(std::function<T(T)>& func, NDArray* target) {
  if (dataType() != DataTypeUtils::fromT<T>())
    THROW_EXCEPTION(
        "NDArray::applyLambda<T> method: wrong template parameter T, its type should be the same as type of this "
        "array!");
  if (dataType() != target->dataType())
    THROW_EXCEPTION("NDArray::applyLambda<T> method: types of this and target array should match!");

  NDArray::preparePrimaryUse({target}, {this});

  auto f = this->bufferAsT<T>();
  auto z = target->bufferAsT<T>();
  const sd::LongType rank = this->rankOf();
  auto xShape = this->shapeOf();
  auto xStride = this->stridesOf();
  auto zStride = target->stridesOf();
  const sd::LongType xBase = this->offset();
  const sd::LongType zBase = target->offset();
  const sd::LongType len = this->lengthOf();

  for (sd::LongType e = 0; e < len; e++) {
    sd::LongType coords[SD_MAX_RANK];
    INDEX2COORDS(e, rank, xShape, coords);
    sd::LongType xOffset, zOffset;
    COORDS2INDEX(rank, xStride, coords, xOffset);
    COORDS2INDEX(rank, zStride, coords, zOffset);
    z[zBase + zOffset] = func(f[xBase + xOffset]);
  }

  NDArray::registerPrimaryUse({target}, {this});
}

template <typename T>
void NDArray::applyIndexedLambda(std::function<T(sd::LongType, T)>& func, NDArray* target) {
  if (dataType() != DataTypeUtils::fromT<T>())
    THROW_EXCEPTION(
        "NDArray::applyIndexedLambda<T> method: wrong template parameter T, its type should be the same as type of "
        "this array!");
  if (dataType() != target->dataType())
    THROW_EXCEPTION("NDArray::applyIndexedLambda<T> method: types of this and target array should match!");

  NDArray::preparePrimaryUse({target}, {this});

  auto f = this->bufferAsT<T>();
  auto z = target->bufferAsT<T>();
  const sd::LongType rank = this->rankOf();
  auto xShape = this->shapeOf();
  auto xStride = this->stridesOf();
  auto zStride = target->stridesOf();
  const sd::LongType xBase = this->offset();
  const sd::LongType zBase = target->offset();
  const sd::LongType len = this->lengthOf();

  for (sd::LongType e = 0; e < len; e++) {
    sd::LongType coords[SD_MAX_RANK];
    INDEX2COORDS(e, rank, xShape, coords);
    sd::LongType xOffset, zOffset;
    COORDS2INDEX(rank, xStride, coords, xOffset);
    COORDS2INDEX(rank, zStride, coords, zOffset);
    z[zBase + zOffset] = func(e, f[xBase + xOffset]);
  }

  NDArray::registerPrimaryUse({target}, {this});
}

template <typename T>
void NDArray::applyPairwiseLambda(NDArray* other, std::function<T(T, T)>& func, NDArray* target) {
  if (dataType() != DataTypeUtils::fromT<T>())
    THROW_EXCEPTION(
        "NDArray::applyPairwiseLambda<T> method: wrong template parameter T, its type should be the same as type of "
        "this array!");
  if (dataType() != other->dataType() || dataType() != target->dataType())
    THROW_EXCEPTION(
        "NDArray::applyPairwiseLambda<T> method: all three arrays (this, other, target) must have the same type!");

  const bool isScalar = other->isScalar();
  if (this->lengthOf() != other->lengthOf() && !this->isScalar() && !isScalar) {
    THROW_EXCEPTION("applyPairwiseLambda requires both operands to have the same shape or one to be a scalar");
  }

  NDArray::preparePrimaryUse({target}, {this, other});

  auto f = this->bufferAsT<T>();
  auto s = other->bufferAsT<T>();
  auto z = target->bufferAsT<T>();
  const sd::LongType rank = this->rankOf();
  auto xShape = this->shapeOf();
  auto xStride = this->stridesOf();
  auto yStride = other->stridesOf();
  auto zStride = target->stridesOf();
  const sd::LongType xBase = this->offset();
  const sd::LongType yBase = other->offset();
  const sd::LongType zBase = target->offset();
  const sd::LongType len = this->lengthOf();

  if (isScalar) {
    const T otherVal = s[other->offset()];
    for (sd::LongType e = 0; e < len; e++) {
      sd::LongType coords[SD_MAX_RANK];
      INDEX2COORDS(e, rank, xShape, coords);
      sd::LongType xOffset, zOffset;
      COORDS2INDEX(rank, xStride, coords, xOffset);
      COORDS2INDEX(rank, zStride, coords, zOffset);
      z[zBase + zOffset] = func(f[xBase + xOffset], otherVal);
    }
  } else {
    for (sd::LongType e = 0; e < len; e++) {
      sd::LongType coords[SD_MAX_RANK];
      INDEX2COORDS(e, rank, xShape, coords);
      sd::LongType xOffset, yOffset, zOffset;
      COORDS2INDEX(rank, xStride, coords, xOffset);
      COORDS2INDEX(rank, yStride, coords, yOffset);
      COORDS2INDEX(rank, zStride, coords, zOffset);
      z[zBase + zOffset] = func(f[xBase + xOffset], s[yBase + yOffset]);
    }
  }

  NDArray::registerPrimaryUse({target}, {this, other});
}

template <typename T>
void NDArray::applyIndexedPairwiseLambda(NDArray* other, std::function<T(sd::LongType, T, T)>& func, NDArray* target) {
  if (dataType() != DataTypeUtils::fromT<T>())
    THROW_EXCEPTION(
        "NDArray::applyIndexedPairwiseLambda<T> method: wrong template parameter T, its type should be the same as "
        "type of this array!");
  if (dataType() != target->dataType())
    THROW_EXCEPTION(
        "NDArray::applyIndexedPairwiseLambda<T> method: types of this and target array should match!");
  if (this->lengthOf() != other->lengthOf()) {
    THROW_EXCEPTION("applyIndexedPairwiseLambda requires both operands to have the same shape");
  }

  NDArray::preparePrimaryUse({target}, {this, other});

  auto f = this->bufferAsT<T>();
  auto s = other->bufferAsT<T>();
  auto z = target->bufferAsT<T>();
  const sd::LongType rank = this->rankOf();
  auto xShape = this->shapeOf();
  auto xStride = this->stridesOf();
  auto yStride = other->stridesOf();
  auto zStride = target->stridesOf();
  const sd::LongType xBase = this->offset();
  const sd::LongType yBase = other->offset();
  const sd::LongType zBase = target->offset();
  const sd::LongType len = this->lengthOf();

  for (sd::LongType e = 0; e < len; e++) {
    sd::LongType coords[SD_MAX_RANK];
    INDEX2COORDS(e, rank, xShape, coords);
    sd::LongType xOffset, yOffset, zOffset;
    COORDS2INDEX(rank, xStride, coords, xOffset);
    COORDS2INDEX(rank, yStride, coords, yOffset);
    COORDS2INDEX(rank, zStride, coords, zOffset);
    z[zBase + zOffset] = func((sd::LongType)e, f[xBase + xOffset], s[yBase + yOffset]);
  }

  NDArray::registerPrimaryUse({target}, {this, other});
}

template <typename T>
void NDArray::applyTriplewiseLambda(NDArray* second, NDArray* third, std::function<T(T, T, T)>& func,
                                    NDArray* target) {
  if (dataType() != DataTypeUtils::fromT<T>())
    THROW_EXCEPTION(
        "NDArray::applyTriplewiseLambda<T> method: wrong template parameter T, its type should be the same as type of "
        "this array!");
  if (dataType() != second->dataType() || dataType() != third->dataType() || dataType() != target->dataType())
    THROW_EXCEPTION(
        "NDArray::applyTriplewiseLambda<T> method: all four arrays (this, second, third, target) should have the "
        "same type!");

  if (this->lengthOf() != second->lengthOf() || this->lengthOf() != third->lengthOf() || !this->isSameShape(second) ||
      !this->isSameShape(third)) {
    std::string errorMessage;
    errorMessage += "applyTriplewiseLambda requires all operands to have the same shape\n";
    errorMessage += "this shape: " + ShapeUtils::shapeAsString(this->shapeInfo()) + "\n";
    errorMessage += "second shape: " + ShapeUtils::shapeAsString(second->shapeInfo()) + "\n";
    errorMessage += "third shape: " + ShapeUtils::shapeAsString(third->shapeInfo()) + "\n";
    errorMessage += "target shape: " + ShapeUtils::shapeAsString(target->shapeInfo()) + "\n";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  NDArray::preparePrimaryUse({target}, {this, second, third});

  auto f = this->bufferAsT<T>();
  auto s = second->bufferAsT<T>();
  auto t = third->bufferAsT<T>();
  auto z = target->bufferAsT<T>();
  const sd::LongType rank = this->rankOf();
  auto xShape = this->shapeOf();
  auto xStride = this->stridesOf();
  auto yStride = second->stridesOf();
  auto wStride = third->stridesOf();
  auto zStride = target->stridesOf();
  const sd::LongType xBase = this->offset();
  const sd::LongType yBase = second->offset();
  const sd::LongType wBase = third->offset();
  const sd::LongType zBase = target->offset();
  const sd::LongType len = this->lengthOf();

  for (sd::LongType e = 0; e < len; e++) {
    sd::LongType coords[SD_MAX_RANK];
    INDEX2COORDS(e, rank, xShape, coords);
    sd::LongType xOffset, yOffset, wOffset, zOffset;
    COORDS2INDEX(rank, xStride, coords, xOffset);
    COORDS2INDEX(rank, yStride, coords, yOffset);
    COORDS2INDEX(rank, wStride, coords, wOffset);
    COORDS2INDEX(rank, zStride, coords, zOffset);
    z[zBase + zOffset] = func(f[xBase + xOffset], s[yBase + yOffset], t[wBase + wOffset]);
  }

  NDArray::registerPrimaryUse({target}, {this, second, third});
}


// Instantiate every lambda method for all common types via the framework type-iteration macros
// (ITERATE_LIST + SD_COMMON_TYPES), the same idiom used in DataBuffer.cpp / NDArrayFactory.cpp --
// not a hand-maintained per-type list. GET_SECOND(T) yields the C++ type for each (ENUM, ctype) tuple.
#define INSTANTIATE_LAMBDA_METHODS(T) template SD_LIB_EXPORT void NDArray::applyLambda(std::function<GET_SECOND(T)(GET_SECOND(T))>& func, NDArray* target);
ITERATE_LIST((SD_COMMON_TYPES), INSTANTIATE_LAMBDA_METHODS)
#undef INSTANTIATE_LAMBDA_METHODS

#define INSTANTIATE_LAMBDA_METHODS_INDEXED(T) template SD_LIB_EXPORT void NDArray::applyIndexedLambda(std::function<GET_SECOND(T)(sd::LongType, GET_SECOND(T))>& func, NDArray* target);
ITERATE_LIST((SD_COMMON_TYPES), INSTANTIATE_LAMBDA_METHODS_INDEXED)
#undef INSTANTIATE_LAMBDA_METHODS_INDEXED

#define INSTANTIATE_LAMBDA_METHODS_PAIRWISE(T) template SD_LIB_EXPORT void NDArray::applyPairwiseLambda(NDArray* other, std::function<GET_SECOND(T)(GET_SECOND(T), GET_SECOND(T))>& func, NDArray* target);
ITERATE_LIST((SD_COMMON_TYPES), INSTANTIATE_LAMBDA_METHODS_PAIRWISE)
#undef INSTANTIATE_LAMBDA_METHODS_PAIRWISE

#define INSTANTIATE_LAMBDA_METHODS_INDEX_PAIR(T) template SD_LIB_EXPORT void NDArray::applyIndexedPairwiseLambda(NDArray* other, std::function<GET_SECOND(T)(sd::LongType, GET_SECOND(T), GET_SECOND(T))>& func, NDArray* target);
ITERATE_LIST((SD_COMMON_TYPES), INSTANTIATE_LAMBDA_METHODS_INDEX_PAIR)
#undef INSTANTIATE_LAMBDA_METHODS_INDEX_PAIR

#define INSTANTIATE_LAMBDA_METHODS_TRIPLE(T) template SD_LIB_EXPORT void NDArray::applyTriplewiseLambda(NDArray* second, NDArray* third, std::function<GET_SECOND(T)(GET_SECOND(T), GET_SECOND(T), GET_SECOND(T))>& func, NDArray* target);
ITERATE_LIST((SD_COMMON_TYPES), INSTANTIATE_LAMBDA_METHODS_TRIPLE)
#undef INSTANTIATE_LAMBDA_METHODS_TRIPLE
} // namespace sd
