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
void scatter(sd::LaunchContext* context, pairwise::Ops op, NDArray& indices, NDArray& updates,
             NDArray& output, const bool lock) {
  const int outRank = output.rankOf();
  const int indRank = indices.rankOf();
  const int updRank = updates.rankOf();
  const sd::LongType indLen = indices.lengthOf();

  if (outRank == 1) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        sd::LongType idx = indices.e<sd::LongType>(i);
        NDArray *out = output({idx, idx + 1});
        NDArray updateE = updates.e(i);
        out->applyPairwiseTransform(op, &updateE);
        delete out;
      }
    };

    samediff::Threads::parallel_tad(func, 0, indLen, 1, lock ? 1 : sd::Environment::getInstance().maxThreads());
  } else {  // outRank > 1

    int sizeOfDims = indRank;
    if (outRank == updRank && indices.isVector()) sizeOfDims = 1;

    std::vector<sd::LongType > dimsToExcludeUpd(sizeOfDims);
    std::iota(dimsToExcludeUpd.begin(), dimsToExcludeUpd.end(), 0);

    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        NDArray *outSubArr = output(indices.e<sd::LongType>(i), std::vector<sd::LongType >({0}));
        NDArray *updSubArr = updates(i, dimsToExcludeUpd);
        outSubArr->applyPairwiseTransform(op, updSubArr);
        delete outSubArr;
        delete updSubArr;
      }
    };

    samediff::Threads::parallel_tad(func, 0, indLen, 1, lock ? 1 : sd::Environment::getInstance().maxThreads());
  }
}

///////////////////////////////////////////////////////////////////
// Direct buffer scatterND — matches the CUDA implementation's approach.
// Uses raw buffer access with coordinate math instead of subarray views
// and applyPairwiseTransform, which crash due to view lifetime issues.
template <typename X, typename Y>
static void scatterND_(pairwise::Ops op, NDArray& indices, NDArray& updates,
                       NDArray& output, const bool lock) {
  const auto x = indices.bufferAsT<X>();           // indices buffer
  const auto y = updates.bufferAsT<Y>();            // updates buffer
  auto z = output.bufferAsT<Y>();                   // output buffer

  const auto xShapeInfo = indices.shapeInfo();
  const auto yShapeInfo = updates.shapeInfo();
  const auto zShapeInfo = output.shapeInfo();

  const int xRank = indices.rankOf();
  const int yRank = updates.rankOf();
  const int zRank = output.rankOf();
  const LongType xLastDim = indices.sizeAt(-1);
  const LongType yLen = updates.lengthOf();

  auto func = PRAGMA_THREADS_FOR {
    LongType yCoords[SD_MAX_RANK];
    LongType zCoords[SD_MAX_RANK];

    for (auto i = start; i < stop; i++) {
      // Convert linear update index to multi-dimensional coordinates
      INDEX2COORDS(i, yRank, shape::shapeOf(yShapeInfo), yCoords);
      LongType yOffset;
      COORDS2INDEX(yRank, shape::stride(yShapeInfo), yCoords, yOffset);

      // Read index values from indices array to determine output position.
      // For each dimension in the last axis of indices, read the target
      // coordinate in the output array.
      bool validIndex = true;
      for (LongType j = 0; j < xLastDim; ++j) {
        // Build indices coordinate: same leading dims as y, last dim = j
        LongType xCoords[SD_MAX_RANK];
        for (int d = 0; d < xRank - 1; d++) {
          xCoords[d] = yCoords[d];
        }
        xCoords[xRank - 1] = j;
        LongType xOffset;
        COORDS2INDEX(xRank, shape::stride(xShapeInfo), xCoords, xOffset);
        zCoords[j] = x[xOffset];

        // Bounds check
        if (zCoords[j] < 0 || zCoords[j] >= shape::shapeOf(zShapeInfo)[j]) {
          validIndex = false;
          break;
        }
      }

      if (!validIndex) continue;

      // Fill remaining z coordinates from trailing y coordinates
      for (LongType j = xLastDim; j < zRank; ++j) {
        zCoords[j] = yCoords[yRank - zRank + j];
      }

      // Compute output offset
      LongType zOffset;
      COORDS2INDEX(zRank, shape::stride(zShapeInfo), zCoords, zOffset);

      // Apply the operation
      switch (op) {
        case pairwise::Add:            z[zOffset] += y[yOffset]; break;
        case pairwise::Subtract:       z[zOffset] -= y[yOffset]; break;
        case pairwise::Multiply:       z[zOffset] *= y[yOffset]; break;
        case pairwise::Divide:         z[zOffset] /= y[yOffset]; break;
        case pairwise::ReverseSubtract: z[zOffset] = y[yOffset] - z[zOffset]; break;
        case pairwise::ReverseDivide:  z[zOffset] = y[yOffset] / z[zOffset]; break;
        case pairwise::CopyPws:        z[zOffset] = y[yOffset]; break;
        case pairwise::MaxPairwise:    z[zOffset] = sd::math::sd_max(z[zOffset], y[yOffset]); break;
        case pairwise::MinPairwise:    z[zOffset] = sd::math::sd_min(z[zOffset], y[yOffset]); break;
        default: break;
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, yLen, 1, lock ? 1 : sd::Environment::getInstance().maxThreads());
}

void scatterND(sd::LaunchContext* context, pairwise::Ops op, NDArray& indices, NDArray& updates,
               NDArray& output, const bool lock) {
  BUILD_DOUBLE_SELECTOR(indices.dataType(), updates.dataType(), scatterND_,
                        (op, indices, updates, output, lock), SD_INTEGER_TYPES, SD_COMMON_TYPES);
}

void scatterForLoss(sd::LaunchContext* context, NDArray& indices, NDArray& updates, NDArray& output,
                    const bool calcGrad) {
  const sd::LongType indicesLen = indices.lengthOf();
  std::vector<sd::LongType> dim = {-1};
  std::vector<sd::LongType > *dimsToExclude = ShapeUtils::evalDimsToExclude(updates.rankOf(), dim.size(),dim.data());

  if (!calcGrad) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        auto subArr = updates(i, *dimsToExclude);
        auto curr = indices.e<sd::LongType>(i);
        output.p(i, curr);
      }
    };

    samediff::Threads::parallel_for(func, 0, indicesLen);

    delete dimsToExclude;
  } else {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        auto subArr = updates(i, *dimsToExclude);
        auto ind = indices.e<sd::LongType>(i);
        auto curr = subArr->e<sd::LongType>(ind) - 1.;
        subArr->p(ind,curr);
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