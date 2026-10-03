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
//
#include <execution/Threads.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/scatter.h>
#include <system/env_functions.h>

#include <numeric>
#if NOT_EXCLUDED(OP_scatter)
namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
// x - indices, z - input/output
template <typename T>
sd::LongType checkIndices_(NDArray& indices, NDArray& output, const int axis) {
  std::atomic<int64_t> numOfBadIndx{0};

  const auto x = indices.bufferAsT<T>();

  const auto xShapeInfo = indices.shapeInfo();
  const auto zShapeInfo = output.shapeInfo();

  // Cache shape information
  const auto xRank = shape::rank(xShapeInfo);
  const auto* xShape = shape::shapeOf(xShapeInfo);
  const auto* xStride = shape::stride(xShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType xCoords[SD_MAX_RANK];

    for (auto i = start; i < stop; i++) {
      INDEX2COORDS(i, xRank, xShape, xCoords);

      sd::LongType xOffset;
      COORDS2INDEX(xRank, xStride, xCoords, xOffset);

      const sd::LongType currentInd = x[xOffset];

      if (currentInd < 0 || currentInd >= shape::sizeAt(zShapeInfo, axis == -1 ? xCoords[xRank - 1] : axis)) {
        ++numOfBadIndx;
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, indices.lengthOf());

  return numOfBadIndx;
}

///////////////////////////////////////////////////////////////////
sd::LongType checkIndices(sd::LaunchContext* context, NDArray& indices, NDArray& output, const int axis) {
  BUILD_SINGLE_SELECTOR(indices.dataType(), return checkIndices_, (indices, output, axis), SD_INTEGER_TYPES);
}

///////////////////////////////////////////////////////////////////
// Applies update y to output element z.
template <typename Y>
static SD_INLINE void applyScatterUpdate(const pairwise::Ops op, Y& z, const Y y) {
  switch (op) {
    case pairwise::Add:
      z += y;
      break;
    case pairwise::Subtract:
      z -= y;
      break;
    case pairwise::Multiply:
      z *= y;
      break;
    case pairwise::Divide:
      z /= y;
      break;
    case pairwise::ReverseSubtract:
      z = y - z;
      break;
    case pairwise::ReverseDivide:
      z = y / z;
      break;
    case pairwise::CopyPws:
      z = y;
      break;
    case pairwise::MaxPairwise:
      z = sd::math::sd_max<Y>(z, y);
      break;
    case pairwise::MinPairwise:
      z = sd::math::sd_min<Y>(z, y);
      break;
    default:
      break;
  }
}

///////////////////////////////////////////////////////////////////
// scatter: index k, the k-th element of indices in logical order, names slice indices[k] of output along its
// first dimension (numSlices of them, sliceLen elements each); update slice k is the k-th run of sliceLen
// elements of updates in logical order, and element p of a slice is its p-th in logical order. That pairing
// serves every updates layout the ops accept: indices.shape + output.shape[1:], [indices.length] +
// output.shape[1:] for vector indices, and indices' shape for a vector output. Indices outside
// [0, numSlices) are skipped.
//
// Updates sharing a destination apply one after another in index order: when an index repeats (or lock is
// set), each thread owns element p of every slice and walks the indices in turn. Otherwise the updates are
// independent and the threads split them.
template <typename X, typename Y>
static void scatter_(pairwise::Ops op, NDArray& indices, NDArray& updates, NDArray& output, const bool lock) {
  const auto x = indices.bufferAsT<X>();
  const auto y = updates.bufferAsT<Y>();
  auto z = output.bufferAsT<Y>();
  const auto xShapeInfo = indices.shapeInfo();
  const auto yShapeInfo = updates.shapeInfo();
  const auto zShapeInfo = output.shapeInfo();
  const int xRank = indices.rankOf();
  const int yRank = updates.rankOf();
  const int zRank = output.rankOf();
  const LongType xLen = indices.lengthOf();
  const LongType numSlices = zRank == 0 ? 1 : output.sizeAt(0);
  const LongType sliceLen = output.lengthOf() / numSlices;

  bool ordered = lock;
  if (!ordered) {
    std::vector<bool> named(numSlices, false);
    LongType coords[SD_MAX_RANK];
    for (LongType k = 0; k < xLen && !ordered; ++k) {
      LongType xOffset;
      INDEX2COORDS(k, xRank, shape::shapeOf(xShapeInfo), coords);
      COORDS2INDEX(xRank, shape::stride(xShapeInfo), coords, xOffset);
      const auto slice = static_cast<LongType>(x[xOffset]);
      if (slice < 0 || slice >= numSlices) continue;
      ordered = named[slice];
      named[slice] = true;
    }
  }

  if (ordered) {
    auto func = PRAGMA_THREADS_FOR {
      LongType coords[SD_MAX_RANK];
      for (auto p = start; p < stop; p++) {
        for (LongType k = 0; k < xLen; ++k) {
          LongType xOffset, yOffset, zOffset;
          INDEX2COORDS(k, xRank, shape::shapeOf(xShapeInfo), coords);
          COORDS2INDEX(xRank, shape::stride(xShapeInfo), coords, xOffset);
          const auto slice = static_cast<LongType>(x[xOffset]);
          if (slice < 0 || slice >= numSlices) continue;

          INDEX2COORDS(k * sliceLen + p, yRank, shape::shapeOf(yShapeInfo), coords);
          COORDS2INDEX(yRank, shape::stride(yShapeInfo), coords, yOffset);
          INDEX2COORDS(slice * sliceLen + p, zRank, shape::shapeOf(zShapeInfo), coords);
          COORDS2INDEX(zRank, shape::stride(zShapeInfo), coords, zOffset);
          applyScatterUpdate<Y>(op, z[zOffset], y[yOffset]);
        }
      }
    };
    samediff::Threads::parallel_for(func, 0, sliceLen);
  } else {
    auto func = PRAGMA_THREADS_FOR {
      LongType coords[SD_MAX_RANK];
      for (auto i = start; i < stop; i++) {
        LongType xOffset, yOffset, zOffset;
        INDEX2COORDS(i / sliceLen, xRank, shape::shapeOf(xShapeInfo), coords);
        COORDS2INDEX(xRank, shape::stride(xShapeInfo), coords, xOffset);
        const auto slice = static_cast<LongType>(x[xOffset]);
        if (slice < 0 || slice >= numSlices) continue;

        INDEX2COORDS(i, yRank, shape::shapeOf(yShapeInfo), coords);
        COORDS2INDEX(yRank, shape::stride(yShapeInfo), coords, yOffset);
        INDEX2COORDS(slice * sliceLen + i % sliceLen, zRank, shape::shapeOf(zShapeInfo), coords);
        COORDS2INDEX(zRank, shape::stride(zShapeInfo), coords, zOffset);
        applyScatterUpdate<Y>(op, z[zOffset], y[yOffset]);
      }
    };
    samediff::Threads::parallel_for(func, 0, updates.lengthOf());
  }
}

///////////////////////////////////////////////////////////////////
void scatter(sd::LaunchContext* context, pairwise::Ops op, NDArray& indices, NDArray& updates,
             NDArray& output, const bool lock) {
  if (indices.lengthOf() == 0 || updates.lengthOf() == 0 || output.lengthOf() == 0) return;
  // The updates are read in output's type.
  NDArray* castUpdates = updates.dataType() == output.dataType() ? nullptr : updates.cast(output.dataType());
  NDArray& typedUpdates = castUpdates != nullptr ? *castUpdates : updates;
  BUILD_DOUBLE_SELECTOR(indices.dataType(), output.dataType(), scatter_, (op, indices, typedUpdates, output, lock),
                        SD_INTEGER_TYPES, SD_COMMON_TYPES);
  delete castUpdates;
}

///////////////////////////////////////////////////////////////////
// scatterND: index row r, the r-th run of indexLength elements of indices in logical order, names output's
// leading indexLength coordinates, flattened into destination slice d (sliceLen elements each); update slice r
// is the r-th run of sliceLen elements of updates, and element p of a slice is its p-th in logical order.
// Rows with a coordinate out of range are skipped. The destination slice of row r, or -1 when a coordinate
// is out of range:
template <typename X>
static LongType scatterNdDestination(const X* x, const LongType* xShapeInfo, const LongType* zShapeInfo,
                                     const LongType indexLength, const LongType row, LongType* coords) {
  const int xRank = shape::rank(xShapeInfo);
  const LongType* zShape = shape::shapeOf(zShapeInfo);
  LongType destination = 0;
  for (LongType j = 0; j < indexLength; ++j) {
    LongType xOffset;
    INDEX2COORDS(row * indexLength + j, xRank, shape::shapeOf(xShapeInfo), coords);
    COORDS2INDEX(xRank, shape::stride(xShapeInfo), coords, xOffset);
    const auto index = static_cast<LongType>(x[xOffset]);
    if (index < 0 || index >= zShape[j]) return -1;
    destination = destination * zShape[j] + index;
  }
  return destination;
}

// As scatter_: repeated destinations (or lock) give each thread element p of every slice to walk the rows in
// order; otherwise the threads split the updates.
template <typename X, typename Y>
static void scatterND_(pairwise::Ops op, NDArray& indices, NDArray& updates, NDArray& output, const bool lock) {
  const auto x = indices.bufferAsT<X>();
  const auto y = updates.bufferAsT<Y>();
  auto z = output.bufferAsT<Y>();
  const auto xShapeInfo = indices.shapeInfo();
  const auto yShapeInfo = updates.shapeInfo();
  const auto zShapeInfo = output.shapeInfo();
  const int yRank = updates.rankOf();
  const int zRank = output.rankOf();
  const LongType indexLength = indices.sizeAt(-1);
  const LongType rows = indices.lengthOf() / indexLength;
  LongType numDestinations = 1;
  for (LongType j = 0; j < indexLength; ++j) numDestinations *= output.sizeAt(j);
  const LongType sliceLen = output.lengthOf() / numDestinations;

  bool ordered = lock;
  if (!ordered) {
    std::vector<bool> named(numDestinations, false);
    LongType coords[SD_MAX_RANK];
    for (LongType r = 0; r < rows && !ordered; ++r) {
      const LongType destination = scatterNdDestination<X>(x, xShapeInfo, zShapeInfo, indexLength, r, coords);
      if (destination < 0) continue;
      ordered = named[destination];
      named[destination] = true;
    }
  }

  if (ordered) {
    auto func = PRAGMA_THREADS_FOR {
      LongType coords[SD_MAX_RANK];
      for (auto p = start; p < stop; p++) {
        for (LongType r = 0; r < rows; ++r) {
          const LongType destination = scatterNdDestination<X>(x, xShapeInfo, zShapeInfo, indexLength, r, coords);
          if (destination < 0) continue;

          LongType yOffset, zOffset;
          INDEX2COORDS(r * sliceLen + p, yRank, shape::shapeOf(yShapeInfo), coords);
          COORDS2INDEX(yRank, shape::stride(yShapeInfo), coords, yOffset);
          INDEX2COORDS(destination * sliceLen + p, zRank, shape::shapeOf(zShapeInfo), coords);
          COORDS2INDEX(zRank, shape::stride(zShapeInfo), coords, zOffset);
          applyScatterUpdate<Y>(op, z[zOffset], y[yOffset]);
        }
      }
    };
    samediff::Threads::parallel_for(func, 0, sliceLen);
  } else {
    auto func = PRAGMA_THREADS_FOR {
      LongType coords[SD_MAX_RANK];
      for (auto i = start; i < stop; i++) {
        const LongType destination =
            scatterNdDestination<X>(x, xShapeInfo, zShapeInfo, indexLength, i / sliceLen, coords);
        if (destination < 0) continue;

        LongType yOffset, zOffset;
        INDEX2COORDS(i, yRank, shape::shapeOf(yShapeInfo), coords);
        COORDS2INDEX(yRank, shape::stride(yShapeInfo), coords, yOffset);
        INDEX2COORDS(destination * sliceLen + i % sliceLen, zRank, shape::shapeOf(zShapeInfo), coords);
        COORDS2INDEX(zRank, shape::stride(zShapeInfo), coords, zOffset);
        applyScatterUpdate<Y>(op, z[zOffset], y[yOffset]);
      }
    };
    samediff::Threads::parallel_for(func, 0, updates.lengthOf());
  }
}

///////////////////////////////////////////////////////////////////
void scatterND(sd::LaunchContext* context, pairwise::Ops op, NDArray& indices, NDArray& updates,
               NDArray& output, const bool lock) {
  if (indices.lengthOf() == 0 || updates.lengthOf() == 0 || output.lengthOf() == 0) return;
  // The updates are read in output's type.
  NDArray* castUpdates = updates.dataType() == output.dataType() ? nullptr : updates.cast(output.dataType());
  NDArray& typedUpdates = castUpdates != nullptr ? *castUpdates : updates;
  BUILD_DOUBLE_SELECTOR(indices.dataType(), output.dataType(), scatterND_, (op, indices, typedUpdates, output, lock),
                        SD_INTEGER_TYPES, SD_COMMON_TYPES);
  delete castUpdates;
}

void scatterForLoss(sd::LaunchContext* context, NDArray& indices, NDArray& updates, NDArray& output,
                    const bool calcGrad) {
  const sd::LongType indicesLen = indices.lengthOf();
  // evalDimsToExclude returns all dims NOT in the given list.
  // We pass the last dimension so that the returned set = {0, 1, ..., rank-2},
  // i.e. the "batch" dimensions that we enumerate over when slicing rows of updates.
  // (Passing -1 as a sentinel would return ALL dims {0,...,rank-1}, giving scalars instead of rows.)
  std::vector<sd::LongType> dim = {updates.rankOf() - 1};
  std::vector<sd::LongType > *dimsToExclude = ShapeUtils::evalDimsToExclude(updates.rankOf(), dim.size(),dim.data());

  if (!calcGrad) {
    // Forward pass: output[i] = updates[i, indices[i]]
    // i.e., gather the log-softmax value at the label index for each example.
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        NDArray* subArr = updates(i, *dimsToExclude);
        auto curr = indices.e<sd::LongType>(i);
        output.p(i, subArr->e<double>(curr));
        delete subArr;
      }
    };

    samediff::Threads::parallel_for(func, 0, indicesLen);

    delete dimsToExclude;
  } else {
    // Gradient path: updates[i, indices[i]] -= 1
    // (matches CUDA: y[yOffset] -= 1.f)
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        auto subArr = updates(i, *dimsToExclude);
        auto ind = indices.e<sd::LongType>(i);
        // Read as double (not LongType) to avoid truncating float softmax values.
        auto curr = subArr->e<double>(ind) - 1.;
        subArr->p(ind, curr);
        delete subArr;
      }
    };

    samediff::Threads::parallel_for(func, 0, indicesLen);
    delete dimsToExclude;
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif