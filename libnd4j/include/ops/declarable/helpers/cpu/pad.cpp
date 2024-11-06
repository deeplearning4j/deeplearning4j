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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//

#include <helpers/Loops.h>
#include <helpers/LoopsCoordsHelper.h>
#include <ops/declarable/helpers/transforms.h>
#include <system/Environment.h>

#include <type_traits>
#if NOT_EXCLUDED(OP_pad)
namespace sd {
namespace ops {
namespace helpers {

template <typename T, size_t constRank>
static void copy_core_rank(const T* x, T* coreZ, const sd::LongType* xShapes, const sd::LongType* xStrides,
                           const sd::LongType* zStrides, int start, int stop) {
  static_assert(constRank > 1, "implement rank 1 directly");
  size_t loop_count = (stop - start);
  sd::ZipCoordsState<constRank - 1> cst;
  sd::zip_size_t offset = sd::init_coords<constRank - 1>(cst, start, xShapes, xStrides, zStrides);
  auto lastStrideX = xStrides[constRank - 1];
  auto lastStrideZ = zStrides[constRank - 1];
  auto inputLastSize = xShapes[constRank - 1];
  if (lastStrideZ == 1 && lastStrideX == 1) {
    for (auto k = 0; k < (stop - start); k++) {
      auto xPtr = &(x[offset.first]);
      auto zPtr = &(coreZ[offset.second]);
      for (int i = 0; i < inputLastSize; i++) {
        zPtr[i] = xPtr[i];
      }
      offset = sd::inc_coords<constRank - 1>(cst, offset);
    }
  } else {
    for (auto k = 0; k < loop_count; k++) {
      auto xPtr = &(x[offset.first]);
      auto zPtr = &(coreZ[offset.second]);
      for (int i = 0; i < inputLastSize; i++) {
        zPtr[i * lastStrideZ] = xPtr[i * lastStrideX];
      }
      offset = sd::inc_coords<constRank - 1>(cst, offset);
    }
  }
}

template <typename T>
void copy_core_generic(int rank, const T* x, T* coreZ, const sd::LongType* xShapes, const sd::LongType* xStrides,
                       const sd::LongType* zStrides, int start, int stop) {
  auto lastStrideX = xStrides[rank - 1];
  auto lastStrideZ = zStrides[rank - 1];
  auto inputLastSize = xShapes[rank - 1];
  sd::LongType coords[SD_MAX_RANK] = {};
  sd::LongType* ptrCoords = (sd::LongType*)&coords;

  zip_size_t offset = {};
  if (rank > 1) {
    index2coords_C(start, rank - 1, xShapes, ptrCoords);
    offset = offset_from_coords(xStrides, zStrides, ptrCoords, rank - 1);
  }
  if (lastStrideZ == 1 && lastStrideX == 1) {
    for (auto k = 0; k < (stop - start); k++) {
      auto xPtr = &(x[offset.first]);
      auto zPtr = &(coreZ[offset.second]);
      for (int i = 0; i < inputLastSize; i++) {
        zPtr[i] = xPtr[i];
      }
      offset = inc_coords(xShapes, xStrides, zStrides, ptrCoords, offset, rank - 1);
    }
  } else {
    for (auto k = 0; k < (stop - start); k++) {
      auto xPtr = &(x[offset.first]);
      auto zPtr = &(coreZ[offset.second]);
      for (int i = 0; i < inputLastSize; i++) {
        zPtr[i * lastStrideZ] = xPtr[i * lastStrideX];
      }
      offset = inc_coords(xShapes, xStrides, zStrides, ptrCoords, offset, rank - 1);
    }
  }
}
//////////////////////////////////////////////////////////////////////////
template <typename T>
void pad_(const int mode, NDArray& input, NDArray& paddings, NDArray& output, NDArray& padValue) {
  const T* x = input.bufferAsT<T>();
  T* z = output.bufferAsT<T>();

  const sd::LongType* xShape = input.shapeOf();
  const sd::LongType* zShape = output.shapeOf();

  const int rank = input.rankOf();  // both input and output have the same rank
  const int rankMinusOne = rank - 1;
  const auto zLen = output.lengthOf();

  if (mode == 0) {  // CONSTANT case

    T padVal = padValue.e<T>(0);

    auto xShapes = input.shapeOf();
    auto outShapes = output.shapeOf();
    auto xStrides = input.stridesOf();
    auto zStrides = output.stridesOf();
    sd::LongType paddingOffsetCoords[SD_MAX_RANK] = {};
    sd::LongType* ptrPaddingCoords = (sd::LongType*)&paddingOffsetCoords;
    bool all_paddings_zero = true;
    for (int j = 0; j < rank; j++) {
      auto p0 = paddings.e<sd::LongType>(j, 0);
      auto p1 = paddings.e<sd::LongType>(j, 1);
      paddingOffsetCoords[j] = p0;

      all_paddings_zero = all_paddings_zero && (p0 == 0) && (p1 == 0);
    }

    auto paddingOffset = all_paddings_zero ? 0 : sd::offset_from_coords(zStrides, ptrPaddingCoords, rank);

    auto inputLastSize = xShapes[rank - 1];

    // fill everything with padding Value
    if (!all_paddings_zero) output.assign(padVal, true);

    // fill the core from input
    auto coreZ = &(z[paddingOffset]);
    // iterate over core
    auto len = input.lengthOf() / inputLastSize;

    auto func = PRAGMA_THREADS_FOR {
      if (rank == 3) {
        copy_core_rank<T, 3>(x, coreZ, xShapes, xStrides, zStrides, start, stop);
      } else if (rank == 4) {
        copy_core_rank<T, 4>(x, coreZ, xShapes, xStrides, zStrides, start, stop);
      } else if (rank == 5) {
        copy_core_rank<T, 5>(x, coreZ, xShapes, xStrides, zStrides, start, stop);
      } else {
        copy_core_generic(rank, x, coreZ, xShapes, xStrides, zStrides, start, stop);
      }
    };
    // fixed restriction for smaller inputs
    auto numThreads = (zLen > 64 || inputLastSize > 4096) ? sd::Environment::getInstance().maxMasterThreads() : 1;
    samediff::Threads::parallel_tad(func, 0, len, 1, numThreads);

  } else {  // REFLECT and SYMMETRIC cases

    const sd::LongType shift1 = mode == 1 ? 0 : 1;  // REFLECT : SYMMETRIC
    const sd::LongType shift2 = mode == 1 ? 2 : 1;  // REFLECT : SYMMETRIC

    auto func = PRAGMA_THREADS_FOR {
      sd::LongType  zCoords[SD_MAX_RANK], xCoords[SD_MAX_RANK];

      for (auto i = start; i < stop; i++) {
        shape::index2coordsCPU(start, i, output.shapeInfo(), zCoords);
        const auto zOffset = shape::getOffset(output.shapeInfo(), zCoords);

        memcpy(xCoords, zCoords, rank * sizeof(sd::LongType));

        for (int j = rankMinusOne; j >= 0; --j) {
          if (xShape[j] == zShape[j]) continue;

          xCoords[j] =
              zCoords[j] - paddings.e<sd::LongType>(j, 0);  // are ready to fill middle (within input dimension range)

          if (xCoords[j] < 0)
            xCoords[j] = -xCoords[j] - shift1;  // means fill from left
          else if (xCoords[j] >= xShape[j])
            xCoords[j] = 2 * xShape[j] - xCoords[j] - shift2;  // means fill from right
        }

        const auto xOffset = shape::getOffset(input.shapeInfo(), xCoords);
        z[zOffset] = x[xOffset];
      }
    };

    samediff::Threads::parallel_tad(func, 0, zLen);
  }
}



void pad(sd::LaunchContext* context, const int mode, NDArray& input, NDArray& paddings, NDArray& output,
         NDArray& padValue) {
  BUILD_SINGLE_SELECTOR(input.dataType(), pad_, (mode, input, paddings, output, padValue), SD_COMMON_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void mirrorPad_(NDArray& input, NDArray& paddings, NDArray& output, const int mode) {
  // mode:  0 - REFLECT, else - SYMMETRIC
  const int reflBorder = (bool)mode ? 1 : 0;
  const int rank = input.rankOf();
  const sd::LongType outLen = output.lengthOf();

  if (input.isScalar() || input.isVector()) {
    const sd::LongType inLen = input.isScalar() ? 1 : input.lengthOf();
    const auto leftSide = paddings.e<sd::LongType>(0);
    const auto leftSideCorrected = leftSide - reflBorder;
    const sd::LongType len = 2 * (inLen - 1) + leftSide + reflBorder;

    for (int i = 0; i < outLen; ++i) {
      if (i < leftSide)  // left side
        output.p(i, input.e<T>(leftSideCorrected - i));

      else if (i >= leftSide && i < leftSide + inLen)  // middle
        output.p(i, input.e<T>(i - leftSide));

      else  // right side
        output.p(i, input.e<T>(len - i));
    }
  } else {
    auto func = PRAGMA_THREADS_FOR {
      sd::LongType  inIdx[SD_MAX_RANK], outIdx[SD_MAX_RANK];

      for (sd::LongType i = start; i < stop; i++) {
        shape::index2coordsCPU(start, i, output.shapeInfo(), outIdx);

        for (int j = 0; j < rank; ++j) {
          const sd::LongType inLen = input.sizeAt(j);
          const auto leftSide = paddings.e<T>(j, 0);
          const auto leftSideCorrected = leftSide - reflBorder;
          const sd::LongType len = 2 * (inLen - 1) + leftSide + reflBorder;

          if (outIdx[j] < leftSide)  // left side
            inIdx[j] = leftSideCorrected - outIdx[j];

          else if (outIdx[j] >= leftSide && outIdx[j] < leftSide + inLen)  // middle
            inIdx[j] = outIdx[j] - leftSide;

          else  // right side
            inIdx[j] = len - outIdx[j];
        }

        auto outOffset = shape::getOffset(output.shapeInfo(), outIdx);
        auto inOffset = shape::getOffset(input.shapeInfo(), inIdx);
        reinterpret_cast<T*>(output.buffer())[outOffset] = reinterpret_cast<T const*>(input.buffer())[inOffset];
      }
    };

    samediff::Threads::parallel_for(func, 0, outLen);
  }
}

void mirrorPad(sd::LaunchContext* context, NDArray& input, NDArray& paddings, NDArray& output,
               const int mode) {
  BUILD_SINGLE_SELECTOR(input.dataType(), mirrorPad_, (input, paddings, output, mode), SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void mirrorPad_,
                      (NDArray& input, NDArray& paddings, NDArray& output, const int mode),
                      SD_COMMON_TYPES);

////////////////////////////////////////////////////////////////////////


}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif