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
sd::LongType checkIndices_(const NDArray& indices, const NDArray& output, const int axis) {
  std::atomic<int64_t> numOfBadIndx{0};

  const auto x = indices.bufferAsT<T>();

  const auto xShapeInfo = indices.shapeInfo();
  const auto zShapeInfo = output.shapeInfo();

  const auto xRank = indices.rankOf();

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType  xCoords[SD_MAX_RANK];

    for (auto i = start; i < stop; i++) {
      shape::index2coordsCPU(start, i, xShapeInfo, xCoords);

      const sd::LongType currentInd = x[shape::getOffset(xShapeInfo, xCoords)];

      if (currentInd >= shape::sizeAt(zShapeInfo, axis == -1 ? xCoords[xRank - 1] : axis)) {
        printf("checkIndices: out of range element %lld at index %ld \n", currentInd, i);
        ++numOfBadIndx;
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, indices.lengthOf());

  return numOfBadIndx;
}

///////////////////////////////////////////////////////////////////
sd::LongType checkIndices(sd::LaunchContext* context, const NDArray& indices, const NDArray& output, const int axis) {
  BUILD_SINGLE_SELECTOR(indices.dataType(), return checkIndices_, (indices, output, axis), SD_INDEXING_TYPES);
}

///////////////////////////////////////////////////////////////////
void scatter(sd::LaunchContext* context, pairwise::Ops op, const NDArray& indices, const NDArray& updates,
             NDArray& output, const bool lock) {
  const int outRank = output.rankOf();
  const int indRank = indices.rankOf();
  const int updRank = updates.rankOf();
  const sd::LongType indLen = indices.lengthOf();

  if (outRank == 1) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        sd::LongType idx = indices.e<sd::LongType>(i);
        NDArray out = output({idx, idx + 1});

        out.applyPairwiseTransform(op, updates.e(i));
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
        NDArray outSubArr = output(indices.e<sd::LongType>(i), std::vector<sd::LongType >({0}));
        NDArray updSubArr = updates(i, dimsToExcludeUpd);
        outSubArr.applyPairwiseTransform(op, updSubArr);
      }
    };

    samediff::Threads::parallel_tad(func, 0, indLen, 1, lock ? 1 : sd::Environment::getInstance().maxThreads());
  }
}

///////////////////////////////////////////////////////////////////
void scatterND(sd::LaunchContext* context, pairwise::Ops op, const NDArray& indices, const NDArray& updates,
               NDArray& output, const bool lock) {
  const sd::LongType indLen = indices.lengthOf();
  const int outRank = output.rankOf();
  const int indRank = indices.rankOf();
  const sd::LongType indLastDim = indices.sizeAt(-1);

  if (outRank == 1) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        sd::LongType idx = indices.e<sd::LongType>(i);
        NDArray out = output({idx, idx + 1});

        out.applyPairwiseTransform(op, updates.e(i), nullptr);
      }
    };

    samediff::Threads::parallel_tad(func, 0, indLen, 1, lock ? 1 : sd::Environment::getInstance().maxThreads());
  } else {
    std::vector<sd::LongType> dims = {indRank - 1};
    std::vector<sd::LongType > *dimsToExcludeInd = ShapeUtils::evalDimsToExclude(indRank, dims.size(),dims.data());
    std::vector<sd::LongType > dimsToExcludeUpd(indRank - 1);
    std::iota(dimsToExcludeUpd.begin(), dimsToExcludeUpd.end(), 0);

    auto func = PRAGMA_THREADS_FOR {
      std::vector<sd::LongType> idxRangeOut(2 * outRank, 0);

      for (auto i = start; i < stop; i++) {
        NDArray indSubArr = indices(i, *dimsToExcludeInd);
        for (sd::LongType j = 0; j < indLastDim; ++j) {
          idxRangeOut[2 * j] = indSubArr.e<sd::LongType>(j);
          idxRangeOut[2 * j + 1] = idxRangeOut[2 * j] + 1;
        }

        NDArray outSubArr = output(idxRangeOut);
        NDArray updSubArr = updates(i, dimsToExcludeUpd);

        outSubArr.applyPairwiseTransform(op, updSubArr);
      }
    };

    samediff::Threads::parallel_tad(func, 0, indLen / indLastDim, 1,
                                    lock ? 1 : sd::Environment::getInstance().maxThreads());

    delete dimsToExcludeInd;

  }
}

void scatterForLoss(sd::LaunchContext* context, const NDArray& indices, NDArray& updates, NDArray& output,
                    const bool calcGrad) {
  // shapes of indices and output must be the same
  // shape of indices should be the same as updates shape with last dimension excluded
  // for example if updates is {a,b,c} then indices should be {a,b}

  const sd::LongType indicesLen = indices.lengthOf();
  std::vector<sd::LongType> dim = {-1};
  std::vector<sd::LongType > *dimsToExclude = ShapeUtils::evalDimsToExclude(updates.rankOf(), dim.size(),dim.data());

  if (!calcGrad) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        auto subArr = updates(i, *dimsToExclude);
        output.p(i, subArr.e(indices.e<sd::LongType>(i)));
      }
    };

    samediff::Threads::parallel_for(func, 0, indicesLen);

    delete dimsToExclude;
  } else {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        auto subArr = updates(i, *dimsToExclude);
        auto ind = indices.e<sd::LongType>(i);
        subArr.p(ind, subArr.e(ind) - 1.);
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