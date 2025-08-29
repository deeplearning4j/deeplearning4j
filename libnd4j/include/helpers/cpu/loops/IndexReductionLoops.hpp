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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 14.03.2019
//
#include <helpers/Loops.h>

using namespace simdOps;

// Helper function to safely convert index to output type
template<typename Z>
SD_INLINE SD_HOST_DEVICE Z convertIndexToZ(sd::LongType index) {
  if constexpr (any_my_string_v<Z>) {
    // For string types, we can't meaningfully store an index
    // You might want to convert to string representation or use default
    return Z{};  // Default construct empty string
    // Alternative: return Z(std::to_string(index)); if you want string representation
  } else {
    return static_cast<Z>(index);
  }
}

//////////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
SD_LIB_EXPORT void sd::IndexReductionLoops<X, Z>::loopIndexReduce( X* x, const LongType* xShapeInfo, Z* z,
                                                                   const LongType* zShapeInfo,
                                                                   const LongType* tadShapeInfo,
                                                                   const LongType* tadOffsets, void* vextraParams) {
  sd::LoopKind::Kind kindOfLoop = sd::LoopKind::deduceKindOfLoopTadXZ(xShapeInfo, zShapeInfo, tadShapeInfo);

  auto extraParams = reinterpret_cast<X*>(vextraParams);
  const sd::LongType zLen = shape::length(zShapeInfo);
  const sd::LongType tadLen = shape::length(tadShapeInfo);

  const sd::LongType* tadShape = shape::shapeOf(const_cast<sd::LongType*>(tadShapeInfo));
  const sd::LongType* tadStride = shape::stride(const_cast<sd::LongType*>(tadShapeInfo));

  switch (kindOfLoop) {
    //*********************************************//

    //*********************************************//

    //*********************************************//
    case sd::LoopKind::RANK1: {
      auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            auto tad = const_cast<X*>(x) + tadOffsets[i];
            auto indexValue = OpType::startingIndexValue(tad);

            for (sd::LongType i0 = 0; i0 < tadLen; ++i0) {
              functions::indexreduce::IndexValue<X> comp(tad[i0 * tadStride[0]], i0);
              indexValue = OpType::update(indexValue, comp, extraParams);
            }

            z[i] = convertIndexToZ<Z>(indexValue.index);
          }
      };

      samediff::Threads::parallel_tad(func, 0, zLen);
    } break;

      //*********************************************//
    case sd::LoopKind::RANK2: {
      sd::LongType newStride[2];
      shape::updateStrides(2, tadShape, newStride, 'c');

      auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            auto tad = const_cast<X*>(x) + tadOffsets[i];
            auto indexValue = OpType::startingIndexValue(tad);

            for (sd::LongType i0 = 0; i0 < tadShape[0]; ++i0) {
              for (sd::LongType i1 = 0; i1 < tadShape[1]; ++i1) {
                const auto tadOffset = i0 * tadStride[0] + i1 * tadStride[1];
                const auto tadIndex = i0 * newStride[0] + i1;
                functions::indexreduce::IndexValue<X> comp(tad[tadOffset], tadIndex);
                indexValue = OpType::update(indexValue, comp, extraParams);
              }
            }

            z[i] = convertIndexToZ<Z>(indexValue.index);
          }
      };

      samediff::Threads::parallel_tad(func, 0, zLen);
    } break;

      //*********************************************//
    case sd::LoopKind::RANK3: {
      sd::LongType newStride[3];
      shape::updateStrides(3, tadShape, newStride, 'c');

      auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            auto tad = const_cast<X*>(x) + tadOffsets[i];
            auto indexValue = OpType::startingIndexValue(tad);

            for (sd::LongType i0 = 0; i0 < tadShape[0]; ++i0) {
              for (sd::LongType i1 = 0; i1 < tadShape[1]; ++i1) {
                for (sd::LongType i2 = 0; i2 < tadShape[2]; ++i2) {
                  const auto tadOffset = i0 * tadStride[0] + i1 * tadStride[1] + i2 * tadStride[2];
                  const auto tadIndex = i0 * newStride[0] + i1 * newStride[1] + i2;
                  functions::indexreduce::IndexValue<X> comp(tad[tadOffset], tadIndex);
                  indexValue = OpType::update(indexValue, comp, extraParams);
                }
              }
            }

            z[i] = convertIndexToZ<Z>(indexValue.index);
          }
      };

      samediff::Threads::parallel_tad(func, 0, zLen);
    } break;

      //*********************************************//
    case sd::LoopKind::RANK4: {
      sd::LongType newStride[4];
      shape::updateStrides(4, tadShape, newStride, 'c');

      auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            auto tad = const_cast<X*>(x) + tadOffsets[i];
            auto indexValue = OpType::startingIndexValue(tad);

            for (sd::LongType i0 = 0; i0 < tadShape[0]; ++i0) {
              for (sd::LongType i1 = 0; i1 < tadShape[1]; ++i1) {
                for (sd::LongType i2 = 0; i2 < tadShape[2]; ++i2) {
                  for (sd::LongType i3 = 0; i3 < tadShape[3]; ++i3) {
                    const auto tadOffset = i0 * tadStride[0] + i1 * tadStride[1] + i2 * tadStride[2] + i3 * tadStride[3];
                    const auto tadIndex = i0 * newStride[0] + i1 * newStride[1] + i2 * newStride[2] + i3;
                    functions::indexreduce::IndexValue<X> comp(tad[tadOffset], tadIndex);
                    indexValue = OpType::update(indexValue, comp, extraParams);
                  }
                }
              }
            }

            z[i] = convertIndexToZ<Z>(indexValue.index);
          }
      };

      samediff::Threads::parallel_tad(func, 0, zLen);
    } break;

      //*********************************************//
    case sd::LoopKind::RANK5: {
      sd::LongType newStride[5];
      shape::updateStrides(5, tadShape, newStride, 'c');

      auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            auto tad = const_cast<X*>(x) + tadOffsets[i];
            auto indexValue = OpType::startingIndexValue(tad);

            for (sd::LongType i0 = 0; i0 < tadShape[0]; ++i0) {
              for (sd::LongType i1 = 0; i1 < tadShape[1]; ++i1) {
                for (sd::LongType i2 = 0; i2 < tadShape[2]; ++i2) {
                  for (sd::LongType i3 = 0; i3 < tadShape[3]; ++i3) {
                    for (sd::LongType i4 = 0; i4 < tadShape[4]; ++i4) {
                      const auto tadOffset = i0 * tadStride[0] + i1 * tadStride[1] + i2 * tadStride[2] +
                                             i3 * tadStride[3] + i4 * tadStride[4];
                      const auto tadIndex =
                          i0 * newStride[0] + i1 * newStride[1] + i2 * newStride[2] + i3 * newStride[3] + i4;
                      functions::indexreduce::IndexValue<X> comp(tad[tadOffset], tadIndex);
                      indexValue = OpType::update(indexValue, comp, extraParams);
                    }
                  }
                }
              }
            }

            z[i] = convertIndexToZ<Z>(indexValue.index);
          }
      };

      samediff::Threads::parallel_tad(func, 0, zLen);
    } break;

      //*********************************************//

      //*********************************************//
    default: {
      sd::LongType tadRank = shape::rank(tadShapeInfo);
      sd::LongType *tadShape = shape::shapeOf(tadShapeInfo);
      sd::LongType *tadStride = shape::stride(tadShapeInfo);
      sd::LongType zRank = shape::rank(zShapeInfo);
      sd::LongType *zShape = shape::shapeOf(zShapeInfo);
      sd::LongType *zStride = shape::stride(zShapeInfo);
      auto func = PRAGMA_THREADS_FOR {
          for (auto i = start; i < stop; i++) {
            auto tad = const_cast<X*>(x) + tadOffsets[i];
            auto indexValue = OpType::startingIndexValue(tad);

            for (sd::LongType j = 0; j < tadLen; j++) {
              LongType coords[SD_MAX_RANK];
              INDEX2COORDS(j, tadRank, tadShape, coords);
              LongType tadOffset;
              COORDS2INDEX(tadRank, tadStride, coords, tadOffset);
              functions::indexreduce::IndexValue<X> comp(tad[tadOffset], j);
              indexValue = OpType::update(indexValue, comp, extraParams);
            }

            LongType coords[SD_MAX_RANK];
            INDEX2COORDS(i, zRank, zShape, coords);
            LongType zOffset;
            COORDS2INDEX(zRank, zStride, coords, zOffset);
            z[zOffset] = convertIndexToZ<Z>(indexValue.index);
          }
      };
      samediff::Threads::parallel_tad(func, 0, zLen);
    }
  }
}

template <typename X, typename Y>
SD_LIB_HIDDEN void sd::IndexReductionLoops<X, Y>::wrapIndexReduce(const int opNum, const void* vx,
                                                                  const sd::LongType* xShapeInfo, void* vz,
                                                                  const sd::LongType* zShapeInfo,
                                                                  const sd::LongType* tadShapeInfo,
                                                                  const sd::LongType* tadOffsets, void* vextraParams) {
  auto x = reinterpret_cast<X*>(const_cast<void *>(vx));
  auto z = reinterpret_cast<Y*>(vz);

  DISPATCH_BY_OPNUM_TT(loopIndexReduce, PARAMS(x, xShapeInfo, z, zShapeInfo, tadShapeInfo, tadOffsets, vextraParams),
                       INDEX_REDUCE_OPS);
}