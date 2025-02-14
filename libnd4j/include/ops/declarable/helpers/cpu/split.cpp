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
//  @author Oleh Semeniv (oleg.semeniv@gmail.com)
//
#include <helpers/Loops.h>
#include <ops/declarable/helpers/transforms.h>
#if NOT_EXCLUDED(OP_split)
namespace sd {
namespace ops {
namespace helpers {



//////////////////////////////////////////////////////////////////////////
template <typename T>
static void split_(NDArray& input, const std::vector<NDArray*>& outArrs, const LongType axis) {
  const sd::LongType numSplits = outArrs.size();
  const auto sizeofT = input.sizeOfT();
  auto xBuff = input.bufferAsT<T>();

  // Cache input shape information
  const sd::LongType* xShapeInfo = input.shapeInfo();
  const sd::LongType* xShape = shape::shapeOf(xShapeInfo);
  const sd::LongType* xStride = shape::stride(xShapeInfo);
  const int xRank = input.rankOf();

  // Fast path 1: Continuous memory case
  bool luckCase1 = ((axis == 0 && input.ordering() == 'c') ||
                    (axis == xRank - 1 && input.ordering() == 'f')) &&
                   input.ews() == 1;

  if (luckCase1) {
    for (sd::LongType i = 0; i < numSplits; ++i) {
      luckCase1 &= outArrs[i]->ordering() == input.ordering() && outArrs[i]->ews() == 1;
      if (!luckCase1) break;
    }
  }

  if (luckCase1) {
    T* x = const_cast<T*>(xBuff);
    for (sd::LongType i = 0; i < numSplits; ++i) {
      const auto memAmountToCopy = outArrs[i]->lengthOf();
      ops::safe_copy(outArrs[i]->bufferAsT<T>(), x, memAmountToCopy);
      x += memAmountToCopy;
    }
    return;
  }

  // Fast path 2: Contiguous along split axis
  const bool isXcontin = input.strideAt(axis) == 1 && input.ordering() == 'c';
  bool areOutsContin = true;
  bool allSameOrder = true;

  if (isXcontin) {
    for (sd::LongType i = 0; i < numSplits; ++i) {
      areOutsContin &= outArrs[i]->strideAt(axis) == 1;
      allSameOrder &= outArrs[i]->ordering() == input.ordering();
      if (!areOutsContin || !allSameOrder) break;
    }
  }

  const bool luckCase2 = isXcontin && areOutsContin && allSameOrder;

  if (luckCase2) {
    const auto xDim = input.sizeAt(axis);

    for (sd::LongType i = 0; i < input.lengthOf() / xDim; ++i) {
      auto x = xBuff + xDim * i;

      for (sd::LongType j = 0; j < numSplits; ++j) {
        const auto zDim = outArrs[j]->sizeAt(axis);
        T* z = outArrs[j]->bufferAsT<T>() + zDim * i;
        ops::safe_copy(z, x, static_cast<size_t>(zDim));

        x += zDim;
      }
    }
    return;
  }

  // General case: Cache output shape data for the first array
  const sd::LongType zDim = outArrs[0]->sizeAt(axis);

  auto func = PRAGMA_THREADS_FOR {
    // Pre-allocate coordinate arrays
    sd::LongType coords[SD_MAX_RANK], temp;

    for (auto i = start; i < stop; i += increment) {
      // Use cached shape data for input coordinates
      INDEX2COORDS(i, xRank, xShape, coords);
      sd::LongType xOffset;
      COORDS2INDEX(xRank, xStride, coords, xOffset);

      // Find target output array
      sd::LongType outArrIdx = 0;
      temp = coords[axis];

      while (coords[axis] >= zDim) {
        coords[axis] -= zDim;
        ++outArrIdx;
      }

      // Get output array and its shape data
      auto outArr = outArrs[outArrIdx];
      const sd::LongType* outShape = shape::shapeOf(outArr->shapeInfo());
      const sd::LongType* outStride = shape::stride(outArr->shapeInfo());
      T* z = outArr->bufferAsT<T>();

      // Calculate output offset using cached shape data
      sd::LongType zOffset;
      COORDS2INDEX(outArr->rankOf(), outStride, coords, zOffset);
      z[zOffset] = xBuff[xOffset];

      coords[axis] = temp;
    }
  };

  samediff::Threads::parallel_for(func, 0, input.lengthOf());
}

void split(sd::LaunchContext* context, NDArray& input, std::vector<NDArray*>& outArrs, const sd::LongType axis) {
  BUILD_SINGLE_SELECTOR(input.dataType(), split_, (input, outArrs, axis), SD_COMMON_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif