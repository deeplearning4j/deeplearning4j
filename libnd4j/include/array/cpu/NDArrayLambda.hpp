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
#pragma once
template <typename T>
SD_LIB_HIDDEN void NDArray::applyTriplewiseLambda(NDArray* second, NDArray* third,
                                                  std::function<T(T, T, T)>& func, NDArray* target) {
  if (dataType() != DataTypeUtils::fromT<T>())
    THROW_EXCEPTION(
        "NDArray::applyTriplewiseLambda<T> method: wrong template parameter T, its type should be the same as type of "
        "this array!");
  if (dataType() != second->dataType() || dataType() != third->dataType() || dataType() != target->dataType())
    THROW_EXCEPTION(
        "NDArray::applyTriplewiseLambda<T> method: bother four arrays (this, second, third, target) should have the "
        "same type !");

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

  auto f = this->bufferAsT<T>();
  auto s = second->bufferAsT<T>();
  auto t = third->bufferAsT<T>();
  auto z = target->bufferAsT<T>();

  // Shared canonical (C-order) index decomposition -- see applyPairwiseLambda for why per-array
  // getOffset(e) mis-pairs operands of differing ordering. (Shapes verified equal above.)
  const sd::LongType lambdaRank = this->rankOf();
  auto lambdaShape   = this->shapeOf();
  auto lambdaXStride = this->stridesOf();
  auto lambdaYStride = second->stridesOf();
  auto lambdaWStride = third->stridesOf();
  auto lambdaZStride = target->stridesOf();
  const sd::LongType lambdaXBase = this->offset();
  const sd::LongType lambdaYBase = second->offset();
  const sd::LongType lambdaWBase = third->offset();
  const sd::LongType lambdaZBase = target->offset();

  auto loop = PRAGMA_THREADS_FOR {
      for (auto e = start; e < stop; e++) {
        sd::LongType lambdaCoords[SD_MAX_RANK];
        INDEX2COORDS(e, lambdaRank, lambdaShape, lambdaCoords);
        sd::LongType tOffset, uOffset, vOffset, zOffset;
        COORDS2INDEX(lambdaRank, lambdaXStride, lambdaCoords, tOffset);
        COORDS2INDEX(lambdaRank, lambdaYStride, lambdaCoords, uOffset);
        COORDS2INDEX(lambdaRank, lambdaWStride, lambdaCoords, vOffset);
        COORDS2INDEX(lambdaRank, lambdaZStride, lambdaCoords, zOffset);
        z[lambdaZBase + zOffset] =
            func(f[lambdaXBase + tOffset], s[lambdaYBase + uOffset], t[lambdaWBase + vOffset]);
      }
  };

  samediff::Threads::parallel_for(loop, 0, _length);
}
#if defined(HAS_DOUBLE)
template SD_LIB_HIDDEN void NDArray::applyTriplewiseLambda(NDArray* second, NDArray* third,
                                                           std::function<double(double, double, double)>& func,
                                                           NDArray* target);
#endif
#if defined(HAS_FLOAT32)
template SD_LIB_HIDDEN void NDArray::applyTriplewiseLambda(NDArray* second, NDArray* third,
                                                           std::function<float(float, float, float)>& func,
                                                           NDArray* target);
#endif
#if defined(HAS_FLOAT16)
template SD_LIB_HIDDEN void NDArray::applyTriplewiseLambda(
    NDArray* second, NDArray* third, std::function<float16(float16, float16, float16)>& func, NDArray* target);
#endif
#if defined(HAS_BFLOAT16)
template SD_LIB_HIDDEN void NDArray::applyTriplewiseLambda(
    NDArray* second, NDArray* third, std::function<bfloat16(bfloat16, bfloat16, bfloat16)>& func,
    NDArray* target);
#endif
#if defined(HAS_INT64)
template SD_LIB_HIDDEN void NDArray::applyTriplewiseLambda(
    NDArray* second, NDArray* third, std::function<sd::LongType(sd::LongType, sd::LongType, sd::LongType)>& func,
    NDArray* target);
#endif
#if defined(HAS_INT32)
template SD_LIB_HIDDEN void NDArray::applyTriplewiseLambda(NDArray* second, NDArray* third,
                                                           std::function<int(int, int, int)>& func,
                                                           NDArray* target);
#endif
#if defined(HAS_INT16)
template SD_LIB_HIDDEN void NDArray::applyTriplewiseLambda(
    NDArray* second, NDArray* third, std::function<int16_t(int16_t, int16_t, int16_t)>& func, NDArray* target);
#endif
#if defined(HAS_INT8)
template SD_LIB_HIDDEN void NDArray::applyTriplewiseLambda(
    NDArray* second, NDArray* third, std::function<char(char, char, char)>& func, NDArray* target);
    template SD_LIB_HIDDEN void NDArray::applyTriplewiseLambda(
    NDArray* second, NDArray* third, std::function<signed char(signed char, signed char, signed char)>& func, NDArray* target);
#endif
#if defined(HAS_UINT8)
template SD_LIB_HIDDEN void NDArray::applyTriplewiseLambda(
    NDArray* second, NDArray* third, std::function<uint8_t(uint8_t, UnsignedChar, UnsignedChar)>& func, NDArray* target);
#endif
#if defined(HAS_UINT16)
template SD_LIB_HIDDEN void NDArray::applyTriplewiseLambda(
    NDArray* second, NDArray* third, std::function<uint16_t(uint16_t, uint16_t, uint16_t)>& func,
    NDArray* target);
#endif
#if defined(HAS_UINT32)
template SD_LIB_HIDDEN void NDArray::applyTriplewiseLambda(
    NDArray* second, NDArray* third, std::function<uint32_t(uint32_t, uint32_t, uint32_t)>& func,
    NDArray* target);
#endif
#if defined(HAS_UNSIGNEDLONG)
template SD_LIB_HIDDEN void NDArray::applyTriplewiseLambda(
    NDArray* second, NDArray* third, std::function<sd::UnsignedLong(sd::UnsignedLong, sd::UnsignedLong, sd::UnsignedLong)>& func,
    NDArray* target);
#endif

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_LIB_HIDDEN void NDArray::applyPairwiseLambda(NDArray* other, std::function<T(T, T)>& func,
                                                NDArray* target) {
  if (dataType() != DataTypeUtils::fromT<T>())
    THROW_EXCEPTION(
        "NDArray::applyPairwiseLambda<T> method: wrong template parameter T, its type should be the same as type of "
        "this array!");
  if (dataType() != other->dataType() || dataType() != target->dataType())
    THROW_EXCEPTION(
        "NDArray::applyPairwiseLambda<T> method: all three arrays (this, other, target) must have the same type !");

  // scalar is broadcastable
  if (this->lengthOf() != other->lengthOf() && !this->isScalar() && !other->isScalar()) {
    THROW_EXCEPTION("applyPairwiseLambda requires both operands to have the same shape");
  }

  auto f = this->bufferAsT<T>();
  auto s = other->bufferAsT<T>();
  auto z = target->bufferAsT<T>();
  auto isTargetOrderEws = !isView() && !target->isView() && this->ordering() == target->ordering() && (shape::strideDescendingCAscendingF(this->shapeInfo()) && shape::strideDescendingCAscendingF(target->shapeInfo()));

  // Shared canonical (C-order) index decomposition for all operands. NDArray::getOffset(e)
  // decomposes the linear index using each array's OWN ordering ('c' vs 'f'), so pairing operands
  // by getOffset(e) silently mis-pairs elements whenever the operands differ in ordering -- the
  // root of corrupted gradients (e.g. relu_bp / thresholdedrelu_bp) when the upstream gradient is
  // non-uniform. Decompose e once over the (this) shape and project onto each array's strides.
  const sd::LongType lambdaRank = this->rankOf();
  auto lambdaXShape  = this->shapeOf();
  auto lambdaXStride = this->stridesOf();
  auto lambdaYStride = other->stridesOf();
  auto lambdaZStride = target->stridesOf();
  const sd::LongType lambdaXBase = this->offset();
  const sd::LongType lambdaYBase = other->offset();
  const sd::LongType lambdaZBase = target->offset();

  if (other->isScalar()) {
    auto otherVal = s[other->getOffset(0)];
    if (isTargetOrderEws) {
      auto loop = PRAGMA_THREADS_FOR {
          for (auto e = start; e < stop; e++) z[e] = func(f[e], otherVal);
      };

      samediff::Threads::parallel_for(loop, 0, _length);
    } else if (f == z) {
      auto loop = PRAGMA_THREADS_FOR {
          for (auto e = start; e < stop; e++) {
            sd::LongType lambdaCoords[SD_MAX_RANK];
            INDEX2COORDS(e, lambdaRank, lambdaXShape, lambdaCoords);
            sd::LongType xOffset;
            COORDS2INDEX(lambdaRank, lambdaXStride, lambdaCoords, xOffset);
            f[lambdaXBase + xOffset] = func(f[lambdaXBase + xOffset], otherVal);
          }
      };

      samediff::Threads::parallel_for(loop, 0, _length);
    } else {
      auto loop = PRAGMA_THREADS_FOR {
          for (auto e = start; e < stop; e++) {
            sd::LongType lambdaCoords[SD_MAX_RANK];
            INDEX2COORDS(e, lambdaRank, lambdaXShape, lambdaCoords);
            sd::LongType xOffset, zOffset;
            COORDS2INDEX(lambdaRank, lambdaXStride, lambdaCoords, xOffset);
            COORDS2INDEX(lambdaRank, lambdaZStride, lambdaCoords, zOffset);
            z[lambdaZBase + zOffset] = func(f[lambdaXBase + xOffset], otherVal);
          }
      };

      samediff::Threads::parallel_for(loop, 0, _length);
    }
    return;
  }

  bool lambdaAllContiguousSameOrder = isTargetOrderEws && this->ordering() == other->ordering() &&
                                      !other->isView() && shape::strideDescendingCAscendingF(other->shapeInfo());
  if (lambdaAllContiguousSameOrder) {
    // All three arrays are contiguous in the same ordering: linear pairing is correct (handles the
    // in-place f == z case too, since z[e] aliases f[e]).
    auto loop = PRAGMA_THREADS_FOR {
        for (auto e = start; e < stop; e++) z[e] = func(f[e], s[e]);
    };

    samediff::Threads::parallel_for(loop, 0, _length);
  } else {
    auto loop = PRAGMA_THREADS_FOR {
        for (auto e = start; e < stop; e++) {
          sd::LongType lambdaCoords[SD_MAX_RANK];
          INDEX2COORDS(e, lambdaRank, lambdaXShape, lambdaCoords);
          sd::LongType xOffset, yOffset, zOffset;
          COORDS2INDEX(lambdaRank, lambdaXStride, lambdaCoords, xOffset);
          COORDS2INDEX(lambdaRank, lambdaYStride, lambdaCoords, yOffset);
          COORDS2INDEX(lambdaRank, lambdaZStride, lambdaCoords, zOffset);
          z[lambdaZBase + zOffset] = func(f[lambdaXBase + xOffset], s[lambdaYBase + yOffset]);
        }
    };

    samediff::Threads::parallel_for(loop, 0, _length);
  }

}

#if defined(HAS_DOUBLE)
template SD_LIB_HIDDEN void NDArray::applyPairwiseLambda(NDArray* other,
                                                         std::function<double(double, double)>& func,
                                                         NDArray* target);
#endif
#if defined(HAS_FLOAT32)
template SD_LIB_HIDDEN void NDArray::applyPairwiseLambda(NDArray* other,
                                                         std::function<float(float, float)>& func,
                                                         NDArray* target);
#endif
#if defined(HAS_FLOAT16)
template SD_LIB_HIDDEN void NDArray::applyPairwiseLambda(NDArray* other,
                                                         std::function<float16(float16, float16)>& func,
                                                         NDArray* target);
#endif
#if defined(HAS_BFLOAT16)
template SD_LIB_HIDDEN void NDArray::applyPairwiseLambda(NDArray* other,
                                                         std::function<bfloat16(bfloat16, bfloat16)>& func,
                                                         NDArray* target);
#endif
#if defined(HAS_INT64)
template SD_LIB_HIDDEN void NDArray::applyPairwiseLambda(
    NDArray* other, std::function<sd::LongType(sd::LongType, sd::LongType)>& func, NDArray* target);
#endif
#if defined(HAS_INT32)
template SD_LIB_HIDDEN void NDArray::applyPairwiseLambda(NDArray* other, std::function<int(int, int)>& func,
                                                         NDArray* target);
#endif
#if defined(HAS_INT16)
template SD_LIB_HIDDEN void NDArray::applyPairwiseLambda(NDArray* other,
                                                         std::function<int16_t(int16_t, int16_t)>& func,
                                                         NDArray* target);
#endif
#if defined(HAS_UINT8)
template SD_LIB_HIDDEN void NDArray::applyPairwiseLambda(NDArray* other,
                                                         std::function<uint8_t(uint8_t, uint8_t)>& func,
                                                         NDArray* target);
#endif
#if defined(HAS_UINT16)
template SD_LIB_HIDDEN void NDArray::applyPairwiseLambda(NDArray* other,
                                                         std::function<uint16_t(uint16_t, uint16_t)>& func,
                                                         NDArray* target);
#endif
#if defined(HAS_UINT32)
template SD_LIB_HIDDEN void NDArray::applyPairwiseLambda(NDArray* other,
                                                         std::function<uint32_t(uint32_t, uint32_t)>& func,
                                                         NDArray* target);
#endif
#if defined(HAS_UNSIGNEDLONG)
template SD_LIB_HIDDEN void NDArray::applyPairwiseLambda(NDArray* other,
                                                         std::function<sd::UnsignedLong(sd::UnsignedLong, sd::UnsignedLong)>& func,
                                                         NDArray* target);
#endif
#if defined(HAS_INT8)
template SD_LIB_HIDDEN void NDArray::applyPairwiseLambda(NDArray* other,
                                                         std::function<int8_t(int8_t, int8_t)>& func,
                                                         NDArray* target);

template SD_LIB_HIDDEN void NDArray::applyPairwiseLambda(NDArray* other,
                                                         std::function<char(char, char)>& func,
                                                         NDArray* target);
#endif
#if defined(HAS_BOOL)
template SD_LIB_HIDDEN void NDArray::applyPairwiseLambda(NDArray* other,
                                                         std::function<bool(bool, bool)>& func, NDArray* target);
#endif

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_LIB_HIDDEN void NDArray::applyLambda(std::function<T(T)>& func, NDArray* target) {
  if (dataType() != DataTypeUtils::fromT<T>())
    THROW_EXCEPTION(
        "NDArray::applyLambda<T> method: wrong template parameter T, its type should be the same as type of this "
        "array!");
  if (dataType() != target->dataType())
    THROW_EXCEPTION("NDArray::applyLambda<T> method: types of this and target array should match !");

  auto f = this->bufferAsT<T>();
  auto z = target->bufferAsT<T>();

  if (f == z) {
    auto loop = PRAGMA_THREADS_FOR {
        for (auto e = start; e < stop; e+= increment) {
          auto xOffset = this->getOffset(e);
          f[xOffset] = func(f[xOffset]);
        }
    };

    samediff::Threads::parallel_for(loop, 0, _length,1);
  } else {
    // Shared canonical (C-order) index decomposition: pairing this and target by their own
    // getOffset(e) yields a transposed copy when the orderings differ. (See applyPairwiseLambda.)
    const sd::LongType lambdaRank = this->rankOf();
    auto lambdaXShape  = this->shapeOf();
    auto lambdaXStride = this->stridesOf();
    auto lambdaZStride = target->stridesOf();
    const sd::LongType lambdaXBase = this->offset();
    const sd::LongType lambdaZBase = target->offset();
    auto loop = PRAGMA_THREADS_FOR {
        for (auto e = start; e < stop; e+= increment) {
          sd::LongType lambdaCoords[SD_MAX_RANK];
          INDEX2COORDS(e, lambdaRank, lambdaXShape, lambdaCoords);
          sd::LongType xOffset, zOffset;
          COORDS2INDEX(lambdaRank, lambdaXStride, lambdaCoords, xOffset);
          COORDS2INDEX(lambdaRank, lambdaZStride, lambdaCoords, zOffset);
          z[lambdaZBase + zOffset] = func(f[lambdaXBase + xOffset]);
        }
    };

    samediff::Threads::parallel_for(loop, 0, _length,1);
  }

}
#if defined(HAS_DOUBLE)
template SD_LIB_HIDDEN void NDArray::applyLambda(std::function<double(double)>& func, NDArray* target);
#endif
#if defined(HAS_FLOAT32)
template SD_LIB_HIDDEN void NDArray::applyLambda(std::function<float(float)>& func, NDArray* target);
#endif
#if defined(HAS_FLOAT16)
template SD_LIB_HIDDEN void NDArray::applyLambda(std::function<float16(float16)>& func, NDArray* target);
#endif
#if defined(HAS_BFLOAT16)
template SD_LIB_HIDDEN void NDArray::applyLambda(std::function<bfloat16(bfloat16)>& func, NDArray* target);
#endif
#if defined(HAS_INT64)
template SD_LIB_HIDDEN void NDArray::applyLambda(std::function<sd::LongType(sd::LongType)>& func,
                                                 NDArray* target);
#endif
#if defined(HAS_INT16)
template SD_LIB_HIDDEN void NDArray::applyLambda(std::function<int16_t(int16_t)>& func, NDArray* target);
#endif
#if defined(HAS_INT32)
template SD_LIB_HIDDEN void NDArray::applyLambda(std::function<int32_t(int32_t)>& func, NDArray* target);
#endif
#if defined(HAS_UINT8)
template SD_LIB_HIDDEN void NDArray::applyLambda(std::function<uint8_t(uint8_t)>& func, NDArray* target);
#endif
#if defined(HAS_UINT16)
template SD_LIB_HIDDEN void NDArray::applyLambda(std::function<uint16_t(uint16_t)>& func, NDArray* target);
#endif
#if defined(HAS_UINT32)
template SD_LIB_HIDDEN void NDArray::applyLambda(std::function<uint32_t(uint32_t)>& func, NDArray* target);
#endif
#if defined(HAS_UNSIGNEDLONG)
template SD_LIB_HIDDEN void NDArray::applyLambda(std::function<sd::UnsignedLong(sd::UnsignedLong)>& func, NDArray* target);
#endif
#if defined(HAS_INT8)
template SD_LIB_HIDDEN void NDArray::applyLambda(std::function<int8_t(int8_t)>& func, NDArray* target);
#endif
#if defined(HAS_BOOL)
template SD_LIB_HIDDEN void NDArray::applyLambda(std::function<bool(bool)>& func, NDArray* target);
#endif

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_LIB_HIDDEN void NDArray::applyIndexedLambda(std::function<T(sd::LongType, T)>& func, NDArray* target) {
  if (dataType() != DataTypeUtils::fromT<T>())
    THROW_EXCEPTION(
        "NDArray::applyIndexedLambda<T> method: wrong template parameter T, its type should be the same as type of "
        "this array!");
  if (dataType() != target->dataType())
    THROW_EXCEPTION("NDArray::applyIndexedLambda<T> method: types of this and target array should match !");

  auto f = this->bufferAsT<T>();
  auto z = target->bufferAsT<T>();

  if (this->ordering() == target->ordering() && (shape::strideDescendingCAscendingF(this->shapeInfo()) && shape::strideDescendingCAscendingF(target->shapeInfo()))) {
    auto loop = PRAGMA_THREADS_FOR {
        for (auto e = start; e < stop; e++) z[e] = func(e, f[e]);
    };

    samediff::Threads::parallel_for(loop, 0, _length);
  } else {
    // Shared canonical (C-order) index decomposition; keeps the func index argument consistent with
    // the contiguous fast path above and avoids transposed pairing for differing orderings.
    const sd::LongType lambdaRank = this->rankOf();
    auto lambdaXShape  = this->shapeOf();
    auto lambdaXStride = this->stridesOf();
    auto lambdaZStride = target->stridesOf();
    const sd::LongType lambdaXBase = this->offset();
    const sd::LongType lambdaZBase = target->offset();
    auto loop = PRAGMA_THREADS_FOR {
        for (auto e = start; e < stop; e++) {
          sd::LongType lambdaCoords[SD_MAX_RANK];
          INDEX2COORDS(e, lambdaRank, lambdaXShape, lambdaCoords);
          sd::LongType xOffset, zOffset;
          COORDS2INDEX(lambdaRank, lambdaXStride, lambdaCoords, xOffset);
          COORDS2INDEX(lambdaRank, lambdaZStride, lambdaCoords, zOffset);
          z[lambdaZBase + zOffset] = func(e, f[lambdaXBase + xOffset]);
        }
    };

    samediff::Threads::parallel_for(loop, 0, _length);
  }
}
#if defined(HAS_DOUBLE)
template SD_LIB_HIDDEN void NDArray::applyIndexedLambda(std::function<double(sd::LongType, double)>& func,
                                                        NDArray* target);
#endif
#if defined(HAS_FLOAT32)
template SD_LIB_HIDDEN void NDArray::applyIndexedLambda(std::function<float(sd::LongType, float)>& func,
                                                        NDArray* target);
#endif
#if defined(HAS_FLOAT16)
template SD_LIB_HIDDEN void NDArray::applyIndexedLambda(std::function<float16(sd::LongType, float16)>& func,
                                                        NDArray* target);
#endif
#if defined(HAS_BFLOAT16)
template SD_LIB_HIDDEN void NDArray::applyIndexedLambda(std::function<bfloat16(sd::LongType, bfloat16)>& func,
                                                        NDArray* target);
#endif
#if defined(HAS_INT64)
template SD_LIB_HIDDEN void NDArray::applyIndexedLambda(
    std::function<sd::LongType(sd::LongType, sd::LongType)>& func, NDArray* target);
#endif
#if defined(HAS_INT32)
template SD_LIB_HIDDEN void NDArray::applyIndexedLambda(std::function<int(sd::LongType, int)>& func,
                                                        NDArray* target);
#endif
#if defined(HAS_INT16)
template SD_LIB_HIDDEN void NDArray::applyIndexedLambda(std::function<int16_t(sd::LongType, int16_t)>& func,
                                                        NDArray* target);
#endif
#if defined(HAS_UINT8)
template SD_LIB_HIDDEN void NDArray::applyIndexedLambda(std::function<uint8_t(sd::LongType, uint8_t)>& func,
                                                        NDArray* target);
#endif
#if defined(HAS_UINT16)
template SD_LIB_HIDDEN void NDArray::applyIndexedLambda(std::function<uint16_t(sd::LongType, uint16_t)>& func,
                                                        NDArray* target);
#endif
#if defined(HAS_UINT32)
template SD_LIB_HIDDEN void NDArray::applyIndexedLambda(std::function<uint32_t(sd::LongType, uint32_t)>& func,
                                                        NDArray* target);
#endif
#if defined(HAS_UNSIGNEDLONG)
template SD_LIB_HIDDEN void NDArray::applyIndexedLambda(std::function<sd::UnsignedLong(sd::LongType, sd::UnsignedLong)>& func,
                                                        NDArray* target);
#endif
#if defined(HAS_INT8)
template SD_LIB_HIDDEN void NDArray::applyIndexedLambda(std::function<int8_t(sd::LongType, int8_t)>& func,
                                                        NDArray* target);
#endif
#if defined(HAS_BOOL)
template SD_LIB_HIDDEN void NDArray::applyIndexedLambda(std::function<bool(sd::LongType, bool)>& func,
                                                        NDArray* target);
#endif

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_LIB_HIDDEN void NDArray::applyIndexedPairwiseLambda(NDArray* other, std::function<T(sd::LongType, T, T)>& func,
                                                       NDArray* target) {
  if (dataType() != DataTypeUtils::fromT<T>())
    THROW_EXCEPTION(
        "NDArray::applyIndexedPairwiseLambda<T> method: wrong template parameter T, its type should be the same as "
        "type of this array!");
  if (dataType() != target->dataType())
    THROW_EXCEPTION(
        "NDArray::applyIndexedPairwiseLambda<T> method: types of this and target array should match !");
  if (this->lengthOf() != other->lengthOf()) {
    sd_printf("applyIndexedPairwiseLambda requires both operands to have the same shape\n", "");
    THROW_EXCEPTION("Shapes mismatch");
  }

  auto f = this->bufferAsT<T>();
  auto s = other->bufferAsT<T>();
  auto z = target->bufferAsT<T>();

  // Shared canonical (C-order) index decomposition -- see applyPairwiseLambda for why per-array
  // getOffset(e) mis-pairs operands of differing ordering.
  const sd::LongType lambdaRank = this->rankOf();
  auto lambdaXShape  = this->shapeOf();
  auto lambdaXStride = this->stridesOf();
  auto lambdaYStride = other->stridesOf();
  auto lambdaZStride = target->stridesOf();
  const sd::LongType lambdaXBase = this->offset();
  const sd::LongType lambdaYBase = other->offset();
  const sd::LongType lambdaZBase = target->offset();

  auto loop = PRAGMA_THREADS_FOR {
      for (auto e = start; e < stop; e++) {
        sd::LongType lambdaCoords[SD_MAX_RANK];
        INDEX2COORDS(e, lambdaRank, lambdaXShape, lambdaCoords);
        sd::LongType xOffset, yOffset, zOffset;
        COORDS2INDEX(lambdaRank, lambdaXStride, lambdaCoords, xOffset);
        COORDS2INDEX(lambdaRank, lambdaYStride, lambdaCoords, yOffset);
        COORDS2INDEX(lambdaRank, lambdaZStride, lambdaCoords, zOffset);
        z[lambdaZBase + zOffset] =
            func((sd::LongType)e, f[lambdaXBase + xOffset], s[lambdaYBase + yOffset]);
      }
  };

  samediff::Threads::parallel_for(loop, 0, _length);
}
#if defined(HAS_DOUBLE)
template SD_LIB_HIDDEN void NDArray::applyIndexedPairwiseLambda(
    NDArray* other, std::function<double(sd::LongType, double, double)>& func, NDArray* target);
#endif
#if defined(HAS_FLOAT32)
template SD_LIB_HIDDEN void NDArray::applyIndexedPairwiseLambda(
    NDArray* other, std::function<float(sd::LongType, float, float)>& func, NDArray* target);
#endif
#if defined(HAS_FLOAT16)
template SD_LIB_HIDDEN void NDArray::applyIndexedPairwiseLambda(
    NDArray* other, std::function<float16(sd::LongType, float16, float16)>& func, NDArray* target);
#endif
#if defined(HAS_BFLOAT16)
template SD_LIB_HIDDEN void NDArray::applyIndexedPairwiseLambda(
    NDArray* other, std::function<bfloat16(sd::LongType, bfloat16, bfloat16)>& func, NDArray* target);
#endif
#if defined(HAS_INT64)
template SD_LIB_HIDDEN void NDArray::applyIndexedPairwiseLambda(
    NDArray* other, std::function<sd::LongType(sd::LongType, sd::LongType, sd::LongType)>& func, NDArray* target);
#endif
#if defined(HAS_INT32)
template SD_LIB_HIDDEN void NDArray::applyIndexedPairwiseLambda(NDArray* other,
                                                                std::function<int(sd::LongType, int, int)>& func,
                                                                NDArray* target);
#endif
#if defined(HAS_INT16)
template SD_LIB_HIDDEN void NDArray::applyIndexedPairwiseLambda(
    NDArray* other, std::function<int16_t(sd::LongType, int16_t, int16_t)>& func, NDArray* target);
#endif
#if defined(HAS_UINT8)
template SD_LIB_HIDDEN void NDArray::applyIndexedPairwiseLambda(
    NDArray* other, std::function<uint8_t(sd::LongType, UnsignedChar, UnsignedChar)>& func, NDArray* target);
#endif
#if defined(HAS_UINT16)
template SD_LIB_HIDDEN void NDArray::applyIndexedPairwiseLambda(
    NDArray* other, std::function<uint16_t(sd::LongType, uint16_t, uint16_t)>& func, NDArray* target);
#endif
#if defined(HAS_UINT32)
template SD_LIB_HIDDEN void NDArray::applyIndexedPairwiseLambda(
    NDArray* other, std::function<uint32_t(sd::LongType, uint32_t, uint32_t)>& func, NDArray* target);
#endif
#if defined(HAS_UNSIGNEDLONG)
template SD_LIB_HIDDEN void NDArray::applyIndexedPairwiseLambda(
    NDArray* other, std::function<sd::UnsignedLong(sd::LongType, sd::UnsignedLong, sd::UnsignedLong)>& func, NDArray* target);
#endif
#if defined(HAS_INT8)
template SD_LIB_HIDDEN void NDArray::applyIndexedPairwiseLambda(
    NDArray* other, std::function<int8_t(sd::LongType, SignedChar, SignedChar)>& func, NDArray* target);
#endif
#if defined(HAS_BOOL)
template SD_LIB_HIDDEN void NDArray::applyIndexedPairwiseLambda(
    NDArray* other, std::function<bool(sd::LongType, bool, bool)>& func, NDArray* target);
#endif
