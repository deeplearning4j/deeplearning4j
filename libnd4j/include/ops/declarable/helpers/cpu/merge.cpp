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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
//
#include <helpers/Loops.h>
#include <ops/declarable/helpers/transforms.h>
#if NOT_EXCLUDED(OP_merge)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void mergeMaxIndex_(const std::vector<NDArray*>& inArrs, NDArray& output) {
  const sd::LongType numArgs = inArrs.size();
  auto x = inArrs[0];

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      X max = -DataTypeUtils::max<X>();
      Z idx = static_cast<Z>(0);

      for (sd::LongType i = 0; i < numArgs; i++) {
        X v = inArrs[i]->t<X>(e);
        if (v > max) {
          max = v;
          idx = static_cast<Z>(i);
        }
      }

      output.r<Z>(e) = static_cast<Z>(idx);
    }
  };

  samediff::Threads::parallel_for(func, 0, x->lengthOf());
}

void mergeMaxIndex(sd::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
  BUILD_DOUBLE_SELECTOR(inArrs[0]->dataType(), output.dataType(), mergeMaxIndex_, (inArrs, output), SD_NUMERIC_TYPES,
                        SD_INDEXING_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void mergeMax_(const std::vector<NDArray*>& inArrs, NDArray& output) {
  const sd::LongType numArgs = inArrs.size();
  auto x = inArrs[0];

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      T max = -DataTypeUtils::max<T>();
      for (sd::LongType i = 0; i < numArgs; i++) {
        T v = inArrs[i]->e<T>(e);
        if (v > max) max = v;
      }
      output.p(e, max);
    }
  };

  samediff::Threads::parallel_for(func, 0, x->lengthOf());
}

void mergeMax(sd::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
  BUILD_SINGLE_SELECTOR(output.dataType(), mergeMax_, (inArrs, output), SD_NUMERIC_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void mergeMaxBp_(const std::vector<NDArray*>& inArrs, std::vector<NDArray*>& outArrs) {
  // outArrs.size() == inArrs.size() - 1
  const sd::LongType numArgs = outArrs.size();
  // last array is gradient
  const auto gradient = inArrs[numArgs]->bufferAsT<T>();
  auto length = inArrs[numArgs]->lengthOf();

  auto gradShape = inArrs[numArgs]->shapeInfo();
  std::vector<bool> vbSameShaepeAndStrides(numArgs);
  std::vector<sd::LongType*> vShapePtrs(numArgs);
  std::vector<sd::LongType*> vStridePtrs(numArgs);
  std::vector<sd::LongType> vRanks(numArgs);
  for (int i = 0; i < numArgs; ++i) {
    vbSameShaepeAndStrides[i] = shape::haveSameShapeAndStrides(gradShape, inArrs[i]->shapeInfo());
    vShapePtrs[i] = shape::shapeOf(inArrs[i]->shapeInfo());
    vStridePtrs[i] = shape::stride(inArrs[i]->shapeInfo());
    vRanks[i] = shape::rank(inArrs[i]->shapeInfo());
  }


  std::vector<sd::LongType *> outShapePtrs(numArgs);
  std::vector<sd::LongType *> outStridePtrs(numArgs);
  std::vector<sd::LongType> outRanks(numArgs);
  for (int i = 0; i < numArgs; ++i) {
    outShapePtrs[i] = shape::shapeOf(outArrs[i]->shapeInfo());
    outStridePtrs[i] = shape::stride(outArrs[i]->shapeInfo());
    outRanks[i] = shape::rank(outArrs[i]->shapeInfo());
  }

  sd::LongType gradRank = shape::rank(gradShape);
  sd::LongType *gradShapeOf = shape::shapeOf(gradShape);
  sd::LongType *gradStride = shape::stride(gradShape);
  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    for (auto e = start; e < stop; e++) {
      INDEX2COORDS(e, gradRank, gradShapeOf, coords);

      sd::LongType gradOffset;
      COORDS2INDEX(gradRank,gradStride, coords, gradOffset);

      T max = -DataTypeUtils::max<T>();
      sd::LongType nMaxIndex = 0;

      for (sd::LongType i = 0; i < numArgs; i++) {
        sd::LongType xOffset;
        if (vbSameShaepeAndStrides[i]) {
          xOffset = gradOffset;
        } else {
          COORDS2INDEX(vRanks[i],vStridePtrs[i], coords, xOffset);
        }
        const T* v = inArrs[i]->bufferAsT<T>();
        if (v[xOffset] > max) {
          max = v[xOffset];
          nMaxIndex = i;
        }
      }

      sd::LongType zOffset;
      if (vbSameShaepeAndStrides[nMaxIndex]) {
        zOffset = gradOffset;
      } else {
        COORDS2INDEX(outRanks[nMaxIndex],outStridePtrs[nMaxIndex], coords, zOffset);
      }

      T* z = outArrs[nMaxIndex]->bufferAsT<T>();
      z[zOffset] = gradient[gradOffset];
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
  return;
}

void mergeMaxBp(sd::LaunchContext* context, const std::vector<NDArray*>& inArrs, std::vector<NDArray*>& outArrs) {
  BUILD_SINGLE_SELECTOR(outArrs[0]->dataType(), mergeMaxBp_, (inArrs, outArrs), SD_NUMERIC_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void mergeAvg_(const std::vector<NDArray*>& inArrs, NDArray& output) {
  const sd::LongType numArgs = inArrs.size();
  const T factor = static_cast<T>(1.f / numArgs);
  auto x = inArrs[0];

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      T sum = static_cast<T>(0);
      for (sd::LongType i = 0; i < numArgs; i++) {
        T v = inArrs[i]->e<T>(e);
        sum += v;
      }
      output.p<T>(e, sum * factor);
    }
  };

  samediff::Threads::parallel_for(func, 0, x->lengthOf());
}

void mergeAvg(sd::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
  BUILD_SINGLE_SELECTOR(output.dataType(), mergeAvg_, (inArrs, output), SD_NUMERIC_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void mergeAvgBp_(NDArray& gradient, std::vector<NDArray*>& outArrs) {
  const sd::LongType numArgs = outArrs.size();

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      T v = gradient.e<T>(e) / numArgs;

      for (sd::LongType i = 0; i < numArgs; i++) {
        outArrs[i]->p<T>(e, v);
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, gradient.lengthOf());
}

void mergeAvgBp(sd::LaunchContext* context, NDArray& gradient, std::vector<NDArray*>& outArrs) {
  BUILD_SINGLE_SELECTOR(gradient.dataType(), mergeAvgBp_, (gradient, outArrs), SD_NUMERIC_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void mergeAdd_(const std::vector<NDArray*>& inArrs, NDArray& output) {
  const sd::LongType numArgs = inArrs.size();
  auto x = inArrs[0];

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      T sum = (T)0.f;
      for (sd::LongType i = 0; i < numArgs; i++) sum += inArrs[i]->e<T>(e);

      output.p(e, sum);
    }
  };

  samediff::Threads::parallel_for(func, 0, x->lengthOf());
}
void mergeAdd(sd::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
  BUILD_SINGLE_SELECTOR(output.dataType(), mergeAdd_, (inArrs, output), SD_NUMERIC_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void mergeAddBp_(NDArray& gradient, std::vector<NDArray*>& outArrs) {
  const sd::LongType numArgs = outArrs.size();

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      T v = gradient.e<T>(e);

      for (sd::LongType i = 0; i < numArgs; i++) {
        outArrs[i]->p<T>(e, v);
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, gradient.lengthOf());
}

void mergeAddBp(sd::LaunchContext* context, NDArray& gradient, std::vector<NDArray*>& outArrs) {
  BUILD_SINGLE_SELECTOR(gradient.dataType(), mergeAddBp_, (gradient, outArrs), SD_NUMERIC_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif