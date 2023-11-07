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

#include <array/TadPack.h>
#include <exceptions/datatype_exception.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/LoopKind.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/broadcasting.h>
#include <loops/broadcasting_bool.h>
#include <loops/broadcasting_int.h>
#include <loops/indexreduce.h>
#include <loops/pairwise_bool.h>
#include <loops/pairwise_int.h>
#include <loops/pairwise_transform.h>
#include <loops/random.h>
#include <loops/reduce3.h>
#include <loops/reduce_bool.h>
#include <loops/reduce_float.h>
#include <loops/reduce_long.h>
#include <loops/reduce_same.h>
#include <loops/scalar.h>
#include <loops/scalar_bool.h>
#include <loops/scalar_int.h>
#include <loops/summarystatsreduce.h>
#include <loops/transform_any.h>
#include <loops/transform_bool.h>
#include <loops/transform_float.h>
#include <loops/transform_same.h>
#include <loops/transform_strict.h>
#include <types/types.h>
#include <array/DataTypeUtils.h>

#include <vector>

#ifdef _OPENMP
#include <helpers/ConstantTadHelper.h>
#include <omp.h>

#endif

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @param hZ
 * @param hZShapeInfo
 */
void NativeOpExecutioner::execIndexReduceScalar(sd::LaunchContext *lc, int opNum, const void *hX,
                                                const sd::LongType *hXShapeInfo, const void *dX,
                                                const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                                const sd::LongType *hZShapeInfo, void *dZ,
                                                const sd::LongType *dZShapeInfo) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  auto hz = reinterpret_cast<sd::LongType *>(hZ);

  BUILD_DOUBLE_SELECTOR(xType, zType, hz[0] = functions::indexreduce::IndexReduce,
                        ::execScalar(opNum, hX, hXShapeInfo, extraParams), SD_COMMON_TYPES, SD_INDEXING_TYPES);
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @param hZ
 * @param hZShapeInfo
 * @param dimension
 * @param dimensionLength
 */

void NativeOpExecutioner::execIndexReduce(sd::LaunchContext *lc, int opNum, const void *hX,
                                          const sd::LongType *hXShapeInfo, const void *dX,
                                          const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                          const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                          sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo,
                                          const sd::LongType *tadOffsets) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  auto hz = reinterpret_cast<sd::LongType *>(hZ);

  BUILD_DOUBLE_SELECTOR(xType, zType, functions::indexreduce::IndexReduce,
                        ::exec(opNum, hX, hXShapeInfo, extraParams, hz, hZShapeInfo, dimension, dimensionLength,
                               tadShapeInfo, tadOffsets),
                        SD_COMMON_TYPES, SD_INDEXING_TYPES);
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param hY
 * @param hYShapeInfo
 * @param hZ
 * @param hZShapeInfo
 * @param dimension
 * @param dimensionLength
 */

void NativeOpExecutioner::execBroadcast(sd::LaunchContext *lc, int opNum, const void *hX,
                                        const sd::LongType *hXShapeInfo, const void *dX,
                                        const sd::LongType *dXShapeInfo, const void *hY,
                                        const sd::LongType *hYShapeInfo, const void *dY,
                                        const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                        void *dZ, const sd::LongType *dZShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength,
                                        const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets,
                                        const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);


#ifdef SD_EXPERIMENTAL_ENABLED
  BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::broadcast::Broadcast,
                          ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, dimension, dimensionLength,
                                 tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);
#else

  auto loopKind = sd::LoopKind::deduceKindOfLoopBroadcast(hXShapeInfo, hYShapeInfo, hZShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    BUILD_SINGLE_SELECTOR_THRICE(
        xType, functions::broadcast::Broadcast,
        ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo,
               tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ, loopKind, start, stop),
        SD_COMMON_TYPES);
  };

  sd::LongType numTads = 0;

  switch (loopKind) {
    case sd::LoopKind::BROADCAST_SCALAR_X: {
      numTads = shape::length(hXShapeInfo);
    } break;
    case sd::LoopKind::BROADCAST_SCALAR_Y: {
      numTads = shape::length(hYShapeInfo);
    } break;
    case sd::LoopKind::BROADCAST_3D: {
      numTads = shape::sizeAt(hZShapeInfo, static_cast<sd::LongType>(0));
    } break;
    case sd::LoopKind::BROADCAST_4D: {
      numTads = shape::sizeAt(hZShapeInfo, static_cast<sd::LongType>(0)) * shape::sizeAt(hZShapeInfo, static_cast<sd::LongType>(1));
    } break;
    case sd::LoopKind::BROADCAST_5D: {
      numTads = shape::sizeAt(hZShapeInfo, static_cast<sd::LongType>(0)) * shape::sizeAt(hZShapeInfo, static_cast<sd::LongType>(1));
    } break;
    default: {
      auto xLen = shape::length(hXShapeInfo);
      auto yLen = shape::length(hYShapeInfo);
      numTads = xLen / yLen;
    }
  }

  samediff::Threads::parallel_tad(func, 0, numTads);

#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcast(sd::LaunchContext *lc, const int opNum, const void *hX,
                                        const sd::LongType *hXShapeInfo, const void *dX,
                                        const sd::LongType *dXShapeInfo, const void *hY,
                                        const sd::LongType *hYShapeInfo, const void *dY,
                                        const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                        void *dZ, const sd::LongType *dZShapeInfo) {

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

#ifdef SD_EXPERIMENTAL_ENABLED
  BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::broadcast::Broadcast,
                          ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo), SD_COMMON_TYPES,
                          SD_COMMON_TYPES);
#else
  BUILD_SINGLE_SELECTOR_THRICE(xType, functions::broadcast::Broadcast,
                               ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo), SD_COMMON_TYPES);
#endif
}

void NativeOpExecutioner::execInverseBroadcast(
    sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo, const void *dX,
    const sd::LongType *dXShapeInfo, const void *hY, const sd::LongType *hYShapeInfo, const void *dY,
    const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
    const sd::LongType *dZShapeInfo,sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo,
    const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);


  if (!sd::Environment::getInstance().isExperimentalBuild())
    if ((yType != xType && yType != sd::DataType::BOOL) || xType != zType)
      throw sd::datatype_exception::build("NativeOps::execBroadcast both operands must have same data type", xType,
                                          yType);

#ifdef SD_EXPERIMENTAL_ENABLED
  BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::broadcast::Broadcast,
                          ::execInverse(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, dimension,
                                        dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);
#else
  auto func = PRAGMA_THREADS_FOR {
    BUILD_SINGLE_SELECTOR_THRICE(
        xType, functions::broadcast::Broadcast,
        ::execInverse(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, dimension, dimensionLength,
                      tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ, start, stop),
        SD_COMMON_TYPES);
  };

  auto xLen = shape::length(hXShapeInfo);
  auto yLen = shape::length(hYShapeInfo);
  auto numTads = yLen / xLen;

  samediff::Threads::parallel_tad(func, 0, numTads);
#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcastBool(sd::LaunchContext *lc, int opNum, const void *hX,
                                            const sd::LongType *hXShapeInfo, const void *dX,
                                            const sd::LongType *dXShapeInfo, const void *hY,
                                            const sd::LongType *hYShapeInfo, const void *dY,
                                            const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                            void *dZ, const sd::LongType *dZShapeInfo, void *extraParams,
                                            sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo,
                                            const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ,
                                            const sd::LongType *tadOffsetsZ) {

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    BUILD_DOUBLE_SELECTOR(
        xType, zType, functions::broadcast::BroadcastBool,
        ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, extraParams, dimension, dimensionLength,
               tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ, start, stop),
        SD_COMMON_TYPES, SD_BOOL_TYPES);
  };

  auto xLen = shape::length(hXShapeInfo);
  auto yLen = shape::length(hYShapeInfo);
  auto numTads = xLen / yLen;

  samediff::Threads::parallel_tad(func, 0, numTads);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcastBool(sd::LaunchContext *lc, const int opNum, const void *hX,
                                            const sd::LongType *hXShapeInfo, const void *dX,
                                            const sd::LongType *dXShapeInfo, const void *hY,
                                            const sd::LongType *hYShapeInfo, const void *dY,
                                            const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                            void *dZ, const sd::LongType *dZShapeInfo, void *extraParams) {

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  BUILD_DOUBLE_SELECTOR(xType, zType, functions::broadcast::BroadcastBool,
                        ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, extraParams), SD_COMMON_TYPES,
                        SD_BOOL_TYPES);
}

void NativeOpExecutioner::execInverseBroadcastBool(
    sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo, const void *dX,
    const sd::LongType *dXShapeInfo, const void *hY, const sd::LongType *hYShapeInfo, const void *dY,
    const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
    const sd::LongType *dZShapeInfo, void *extraParams,sd::LongType *dimension, sd::LongType dimensionLength,
    const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ,
    const sd::LongType *tadOffsetsZ) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);


  if (!sd::Environment::getInstance().isExperimentalBuild())
    if (yType != xType || sd::DataType::BOOL != zType)
      throw sd::datatype_exception::build("NativeOps::execInverseBroadcastBool both operands must have same data type",
                                          xType, yType);

  auto func = PRAGMA_THREADS_FOR {
    BUILD_DOUBLE_SELECTOR(
        xType, zType, functions::broadcast::BroadcastBool,
        ::execInverse(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, extraParams, dimension, dimensionLength,
                      tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ, start, stop),
        SD_COMMON_TYPES, SD_BOOL_TYPES);
  };

  auto xLen = shape::length(hXShapeInfo);
  auto yLen = shape::length(hYShapeInfo);
  auto numTads = yLen / xLen;

  samediff::Threads::parallel_tad(func, 0, numTads);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcastInt(
    sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo, const void *dX,
    const sd::LongType *dXShapeInfo, const void *hY, const sd::LongType *hYShapeInfo, const void *dY,
    const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
    const sd::LongType *dZShapeInfo,sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo,
    const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);


  if (xType != yType || xType != zType)
    throw sd::datatype_exception::build("NativeOpExecutioner::execBroadcastInt", zType, xType, yType);

  if (!sd::DataTypeUtils::isZ(zType))
    throw sd::datatype_exception::build("NativeOpExecutioner::execBroadcastInt requires integer data type", zType);

  auto func = PRAGMA_THREADS_FOR {
    BUILD_SINGLE_SELECTOR(xType, functions::broadcast::BroadcastInt,
                          ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, dimension, dimensionLength,
                                 tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ, start, stop),
                          SD_INTEGER_TYPES);
  };

  auto xLen = shape::length(hXShapeInfo);
  auto yLen = shape::length(hYShapeInfo);
  auto numTads = xLen / yLen;

  samediff::Threads::parallel_tad(func, 0, numTads);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcastInt(sd::LaunchContext *lc, const int opNum, const void *hX,
                                           const sd::LongType *hXShapeInfo, const void *dX,
                                           const sd::LongType *dXShapeInfo, const void *hY,
                                           const sd::LongType *hYShapeInfo, const void *dY,
                                           const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                           void *dZ, const sd::LongType *dZShapeInfo) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);


  if (xType != yType || xType != zType)
    throw sd::datatype_exception::build("NativeOpExecutioner::execBroadcastInt", zType, xType, yType);

  if (!sd::DataTypeUtils::isZ(zType))
    throw sd::datatype_exception::build("NativeOpExecutioner::execBroadcastInt requires integer data type", zType);

  BUILD_SINGLE_SELECTOR(xType, functions::broadcast::BroadcastInt,
                        ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo), SD_INTEGER_TYPES);
}

void NativeOpExecutioner::execInverseBroadcastInt(
    sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo, const void *dX,
    const sd::LongType *dXShapeInfo, const void *hY, const sd::LongType *hYShapeInfo, const void *dY,
    const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
    const sd::LongType *dZShapeInfo,sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo,
    const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);


  if (xType != yType || xType != zType)
    throw sd::datatype_exception::build("NativeOpExecutioner::execInverseBroadcastInt", zType, xType, yType);

  if (!sd::DataTypeUtils::isZ(zType))
    throw sd::datatype_exception::build("NativeOpExecutioner::execInverseBroadcastInt requires integer data type",
                                        zType);

  auto func = PRAGMA_THREADS_FOR {
    BUILD_SINGLE_SELECTOR(
        xType, functions::broadcast::BroadcastInt,
        ::execInverse(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, dimension, dimensionLength,
                      tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ, start, stop),
        SD_INTEGER_TYPES);
  };

  auto xLen = shape::length(hXShapeInfo);
  auto yLen = shape::length(hYShapeInfo);
  auto numTads = yLen / xLen;

  samediff::Threads::parallel_tad(func, 0, numTads);
}

////////////////////////////////////////////////////////////////////////
bool isViewOf(const void* ptr1, size_t size1, const void* ptr2, size_t size2) {
  uintptr_t start1 = reinterpret_cast<uintptr_t>(ptr1);
  uintptr_t end1 = start1 + size1;

  uintptr_t start2 = reinterpret_cast<uintptr_t>(ptr2);
  uintptr_t end2 = start2 + size2;

  return (start1 >= start2 && start1 < end2) || (end1 > start2 && end1 <= end2) ||
         (start2 >= start1 && start2 < end1) || (end2 > start1 && end2 <= end1);
}
/**
 *
 * @param opNum
 * @param hX
 * @param xStride
 * @param hY
 * @param yStride
 * @param hZ
 * @param resultStride
 * @param extraParams
 * @param n
 */
void NativeOpExecutioner::execPairwiseTransform(sd::LaunchContext *lc, int opNum, const void *hX,
                                                const sd::LongType *hXShapeInfo, const void *dX,
                                                const sd::LongType *dXShapeInfo, const void *hY,
                                                const sd::LongType *hYShapeInfo, const void *dY,
                                                const sd::LongType *dYShapeInfo, void *hZ,
                                                const sd::LongType *hZShapeInfo, void *dZ,
                                                const sd::LongType *dZShapeInfo, void *extraParams) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
#ifdef SD_EXPERIMENTAL_ENABLED
  BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::pairwise_transforms::PairWiseTransform,
                          ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, extraParams),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);
#else
  auto func = PRAGMA_THREADS_FOR {
    BUILD_SINGLE_SELECTOR_THRICE(
        xType, functions::pairwise_transforms::PairWiseTransform,
        ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, extraParams, start, stop), SD_COMMON_TYPES);
  };



  auto zLen = shape::length(hZShapeInfo);
  samediff::Threads::parallel_for(
      func, 0, zLen, 1,
      sd::math::sd_max<int>(1, sd::math::sd_min<int>(zLen / 1024, sd::Environment::getInstance().maxMasterThreads())));



#endif

}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execPairwiseBoolTransform(sd::LaunchContext *lc, int opNum, const void *hX,
                                                    const sd::LongType *hXShapeInfo, const void *dX,
                                                    const sd::LongType *dXShapeInfo, const void *hY,
                                                    const sd::LongType *hYShapeInfo, const void *dY,
                                                    const sd::LongType *dYShapeInfo, void *hZ,
                                                    const sd::LongType *hZShapeInfo, void *dZ,
                                                    const sd::LongType *dZShapeInfo, void *extraParams) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);


  if (xType != yType)
    throw sd::datatype_exception::build("NativeOpExecutioner::execPairwiseBoolTransform", xType, yType);

  if (zType != sd::DataType::BOOL)
    throw sd::datatype_exception::build("NativeOpExecutioner::execPairwiseBoolTransform", sd::DataType::BOOL, zType);

  auto func = PRAGMA_THREADS_FOR {
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::pairwise_transforms::PairWiseBoolTransform,
                          ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, extraParams, start, stop),
                          SD_COMMON_TYPES, SD_BOOL_TYPES);
  };

  auto zLen = shape::length(hZShapeInfo);
  samediff::Threads::parallel_for(
      func, 0, zLen, 1,
      sd::math::sd_max<int>(1, sd::math::sd_min<int>(zLen / 1024, sd::Environment::getInstance().maxMasterThreads())));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execPairwiseIntTransform(sd::LaunchContext *lc, int opNum, const void *hX,
                                                   const sd::LongType *hXShapeInfo, const void *dX,
                                                   const sd::LongType *dXShapeInfo, const void *hY,
                                                   const sd::LongType *hYShapeInfo, const void *dY,
                                                   const sd::LongType *dYShapeInfo, void *hZ,
                                                   const sd::LongType *hZShapeInfo, void *dZ,
                                                   const sd::LongType *dZShapeInfo, void *extraParams) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);


  if (xType != yType || xType != zType)
    throw sd::datatype_exception::build("NativeOpExecutioner::execPairwiseIntTransform", zType, xType, yType);

  if (!sd::DataTypeUtils::isZ(zType))
    throw sd::datatype_exception::build("NativeOpExecutioner::execSPairwiseInt requires integer data type", zType);

  auto func = PRAGMA_THREADS_FOR {
    BUILD_SINGLE_SELECTOR(xType, functions::pairwise_transforms::PairWiseIntTransform,
                          ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, extraParams, start, stop),
                          SD_INTEGER_TYPES);
  };

  auto zLen = shape::length(hZShapeInfo);
  samediff::Threads::parallel_for(
      func, 0, zLen, 1,
      sd::math::sd_max<int>(1, sd::math::sd_min<int>(zLen / 1024, sd::Environment::getInstance().maxMasterThreads())));
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @param hZ
 * @param hZShapeInfo
 */
void NativeOpExecutioner::execReduceFloat(sd::LaunchContext *lc, int opNum, const void *hX,
                                          const sd::LongType *hXShapeInfo, const void *dX,
                                          const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                          const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                          sd::LongType *dimension, sd::LongType dimensionLength) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  //Note here we continue due to numpy compat which
  //expects an output for a reduce.
  // numpy compat: default is 1 for 0 length arrays https://stackoverflow.com/questions/66746566/numpy-explanation-of-numpy-prod

  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::reduce::ReduceFloatFunction,
      ::exec(opNum, lc ? lc->getWorkspace() : nullptr, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo, dimension),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceSame(sd::LaunchContext *lc, int opNum, const void *hX,
                                         const sd::LongType *hXShapeInfo, const void *dX,
                                         const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                         const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                         sd::LongType *dimension, sd::LongType dimensionLength) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  BUILD_SINGLE_SELECTOR(
      xType, functions::reduce::ReduceSameFunction,
      ::exec(opNum, lc ? lc->getWorkspace() : nullptr, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo, dimension),
      SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceBool(sd::LaunchContext *lc, int opNum, const void *hX,
                                         const sd::LongType *hXShapeInfo, const void *dX,
                                         const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                         const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                         sd::LongType *dimension, sd::LongType dimensionLength) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::reduce::ReduceBoolFunction,
      ::exec(opNum, lc ? lc->getWorkspace() : nullptr, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo, dimension),
      SD_COMMON_TYPES, SD_BOOL_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceLong(sd::LaunchContext *lc, int opNum, const void *hX,
                                         const sd::LongType *hXShapeInfo, const void *dX,
                                         const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                         const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                         sd::LongType *dimension, sd::LongType dimensionLength) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::reduce::ReduceLongFunction,
      ::exec(opNum, lc ? lc->getWorkspace() : nullptr, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo, dimension),
      SD_COMMON_TYPES, SD_LONG_TYPES);
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @return
 */
void NativeOpExecutioner::execReduceFloatScalar(sd::LaunchContext *lc, int opNum, const void *hX,
                                                const sd::LongType *hXShapeInfo, const void *dX,
                                                const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                                const sd::LongType *hZShapeInfo, void *dZ,
                                                const sd::LongType *dZShapeInfo) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceFloatFunction,
                        ::execScalar(opNum, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo), SD_COMMON_TYPES,
                        SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceSameScalar(sd::LaunchContext *lc, int opNum, const void *hX,
                                               const sd::LongType *hXShapeInfo, const void *dX,
                                               const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                               const sd::LongType *hZShapeInfo, void *dZ,
                                               const sd::LongType *dZShapeInfo) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  BUILD_SINGLE_SELECTOR(xType, functions::reduce::ReduceSameFunction,
                        ::execScalar(opNum, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo), SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceBoolScalar(sd::LaunchContext *lc, int opNum, const void *hX,
                                               const sd::LongType *hXShapeInfo, const void *dX,
                                               const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                               const sd::LongType *hZShapeInfo, void *dZ,
                                               const sd::LongType *dZShapeInfo) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceBoolFunction,
                        ::execScalar(opNum, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo), SD_COMMON_TYPES,
                        SD_BOOL_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceLongScalar(sd::LaunchContext *lc, int opNum, const void *hX,
                                               const sd::LongType *hXShapeInfo, const void *dX,
                                               const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                               const sd::LongType *hZShapeInfo, void *dZ,
                                               const sd::LongType *dZShapeInfo) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceLongFunction,
                        ::execScalar(opNum, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo), SD_COMMON_TYPES,
                        SD_LONG_TYPES);
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParamsVals
 * @param hY
 * @param hYShapeInfo
 * @param hZ
 * @param hZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void NativeOpExecutioner::execReduce3Scalar(sd::LaunchContext *lc, int opNum, const void *hX,
                                            const sd::LongType *hXShapeInfo, const void *dX,
                                            const sd::LongType *dXShapeInfo, void *extraParamsVals, const void *hY,
                                            const sd::LongType *hYShapeInfo, const void *dY,
                                            const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                            void *dZ, const sd::LongType *dZShapeInfo) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3,
                        ::execScalar(opNum, hX, hXShapeInfo, extraParamsVals, hY, hYShapeInfo, hZ, hZShapeInfo),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParamsVals
 * @param hY
 * @param hYShapeInfo
 * @param hZ
 * @param hZShapeInfo
 */
void NativeOpExecutioner::execReduce3(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                      const void *dX, const sd::LongType *dXShapeInfo, void *extraParamsVals,
                                      const void *hY, const sd::LongType *hYShapeInfo, const void *dY,
                                      const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                      void *dZ, const sd::LongType *dZShapeInfo) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  NativeOpExecutioner::execReduce3Scalar(lc, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParamsVals, hY, hYShapeInfo,
                                         dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                      const void *dX, const sd::LongType *dXShapeInfo, void *extraParamsVals,
                                      const void *hY, const sd::LongType *hYShapeInfo, const void *dY,
                                      const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                      void *dZ, const sd::LongType *dZShapeInfo,sd::LongType *dimension, sd::LongType dimensionLength,
                                      const sd::LongType *xTadOnlyShapeInfo, const sd::LongType *xTadOffsets,
                                      const sd::LongType *yTadOnlyShapeInfo, const sd::LongType *yTadOffsets) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  const auto xLen = shape::length(hXShapeInfo);
  const auto yLen = shape::length(hYShapeInfo);

  sd::TadPack *tadPack;

  if (xLen == yLen) {
    tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);
  } else if (yLen > xLen) {
    tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(hYShapeInfo, dimension, dimensionLength);
  } else {
    tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);
  }

  auto func = PRAGMA_THREADS_FOR {
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3,
                          ::exec(opNum, hX, hXShapeInfo, extraParamsVals, hY, hYShapeInfo, hZ, hZShapeInfo, dimension,
                                 dimensionLength, start, stop),
                          SD_COMMON_TYPES, SD_FLOAT_TYPES);
  };

  samediff::Threads::parallel_tad(func, 0, tadPack->numberOfTads());
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3All(sd::LaunchContext *lc, int opNum, const void *hX,
                                         const sd::LongType *hXShapeInfo, const void *dX,
                                         const sd::LongType *dXShapeInfo, void *extraParamsVals, const void *hY,
                                         const sd::LongType *hYShapeInfo, const void *dY,
                                         const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                         void *dZ, const sd::LongType *dZShapeInfo,sd::LongType *dimension, sd::LongType dimensionLength,
                                         const sd::LongType *xTadShapeInfo, const sd::LongType *xOffsets,
                                         const sd::LongType *yTadShapeInfo, const sd::LongType *yOffsets) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);

  // TODO: make it 2d
  auto func = PRAGMA_THREADS_FOR {
    BUILD_DOUBLE_SELECTOR(
        xType, zType, functions::reduce3::Reduce3,
        ::execAll(opNum, hX, hXShapeInfo, extraParamsVals, hY, hYShapeInfo, hZ, hZShapeInfo, dimension, dimensionLength,
                  xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets, start, stop),
        SD_COMMON_TYPES, SD_FLOAT_TYPES);
  };

  samediff::Threads::parallel_tad(func, 0, tadPack->numberOfTads());
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3TAD(sd::LaunchContext *lc, int opNum, const void *hX,
                                         const sd::LongType *hXShapeInfo, const void *dX,
                                         const sd::LongType *dXShapeInfo, void *extraParamsVals, const void *hY,
                                         const sd::LongType *hYShapeInfo, const void *dY,
                                         const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                         void *dZ, const sd::LongType *dZShapeInfo,sd::LongType *dimension, sd::LongType dimensionLength,
                                         const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets,
                                         const sd::LongType *yTadShapeInfo, const sd::LongType *yTadOffsets) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  const auto xLen = shape::length(hXShapeInfo);
  const auto yLen = shape::length(hYShapeInfo);

  sd::TadPack *tadPack;

  if (xLen == yLen) {
    tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);
  } else if (yLen > xLen) {
    tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(hYShapeInfo, dimension, dimensionLength);
  } else {
    tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);
  }

  auto func = PRAGMA_THREADS_FOR {
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3,
                          ::exec(opNum, hX, hXShapeInfo, extraParamsVals, hY, hYShapeInfo, hZ, hZShapeInfo, dimension,
                                 dimensionLength, tadShapeInfo, tadOffsets, start, stop),
                          SD_COMMON_TYPES, SD_FLOAT_TYPES);
  };

  samediff::Threads::parallel_tad(func, 0, tadPack->numberOfTads());
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param hX
 * @param xStride
 * @param hZ
 * @param resultStride
 * @param scalar
 * @param extraParams
 * @param n
 */
void NativeOpExecutioner::execScalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                     const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                                     const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                     const void *hScalar, const sd::LongType *hScalarShapeInfo, const void *dScalar,
                                     const sd::LongType *dScalarShapeInfo, void *extraParams, bool allowParallelism) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);


#ifdef SD_EXPERIMENTAL_ENABLED
  BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::scalar::ScalarTransform,
                          ::transform(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, hScalar, extraParams), SD_COMMON_TYPES,
                          SD_COMMON_TYPES_ALL);

#else
  if (xType != yType || xType != zType) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execScalar requires both X & Y to have same data type";
    errorMessage += "X data type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += ", Y data type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);
    errorMessage += ", Z data type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());

  }

  auto func = PRAGMA_THREADS_FOR {
    BUILD_SINGLE_SELECTOR_THRICE(
        xType, functions::scalar::ScalarTransform,
        ::transform(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, hScalar, extraParams, start, stop), SD_COMMON_TYPES_ALL);
  };

  auto zLen = shape::length(hZShapeInfo);
  samediff::Threads::parallel_for(
      func, 0, zLen, 1,
      !allowParallelism
      ? 1
      : sd::math::sd_max<int>(
          1, sd::math::sd_min<int>(zLen / 1024, sd::Environment::getInstance().maxMasterThreads())));


#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalar(sd::LaunchContext *lc, int opNum, void const *hX, sd::LongType const *hXShapeInfo,
                                     void const *dX, sd::LongType const *dXShapeInfo, void *extraParams, void *hZ,
                                     sd::LongType const *hZShapeInfo, void *dZ, sd::LongType const *dZShapeInfo,
                                     void const *hScalars, sd::LongType const *hScalarShapeInfo, void const *dScalars,
                                     sd::LongType const *dScalarShapeInfo,sd::LongType *dimension, sd::LongType dimensionLength,
                                     sd::LongType const *tadShapeInfo, sd::LongType const *tadOffsets,
                                     sd::LongType const *tadShapeInfoZ, sd::LongType const *tadOffsetsZ) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

#ifdef SD_EXPERIMENTAL_ENABLED
  BUILD_PAIRWISE_SELECTOR(xType, yType, zType, functions::scalar::ScalarTransform,
                          ::transform(opNum, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo, hScalars, dimension,
                                      dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);
#else
  if (xType != yType || xType != zType) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execScalar requires both X & Y to have same data type";
    errorMessage += "X data type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += ", Y data type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);
    errorMessage += ", Z data type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());

  }
  auto func = PRAGMA_THREADS_FOR {
    BUILD_SINGLE_SELECTOR_THRICE(
        xType, functions::scalar::ScalarTransform,
        ::transform(opNum, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo, hScalars, dimension, dimensionLength,
                    tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ, start, stop),
        SD_COMMON_TYPES_ALL);
  };

  auto yLen = shape::length(hScalarShapeInfo);
  samediff::Threads::parallel_tad(func, 0, yLen, 1,
                                  sd::math::sd_min<int>(yLen, sd::Environment::getInstance().maxMasterThreads()));

#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarBool(sd::LaunchContext *lc, int opNum, const void *hX,
                                         const sd::LongType *hXShapeInfo, const void *dX,
                                         const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                         void *dZ, const sd::LongType *dZShapeInfo, const void *hScalar,
                                         const sd::LongType *hSscalarShapeInfo, const void *dScalar,
                                         const sd::LongType *dSscalarShapeInfo, void *extraParams,
                                         bool allowParallelism) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hSscalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  if (xType != yType || xType != zType) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execScalarBool requires both X & Y to have same data type";
    errorMessage += "X data type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += ", Y data type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);
    errorMessage += ", Z data type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());

  }
  if (zType != sd::DataType::BOOL)
    throw sd::datatype_exception::build("NativeOpExecutioner::execScalarBool", sd::DataType::BOOL, zType);

  auto func = PRAGMA_THREADS_FOR {
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::scalar::ScalarBoolTransform,
                          ::transform(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, hScalar, extraParams, start, stop),
                          SD_COMMON_TYPES_ALL, SD_BOOL_TYPES);
  };

  auto zLen = shape::length(hZShapeInfo);
  samediff::Threads::parallel_for(
      func, 0, zLen, 1,
      !allowParallelism
      ? 1
      : sd::math::sd_max<int>(
          1, sd::math::sd_min<int>(zLen / 1024, sd::Environment::getInstance().maxMasterThreads())));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarBool(
    sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo, const void *dX,
    const sd::LongType *dXShapeInfo, void *extraParams, void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
    const sd::LongType *dZShapeInfo, const void *hScalars, const sd::LongType *hScalarShapeInfo, const void *dScalars,
    const sd::LongType *dScalarShapeInfo,
    sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo,
    const sd::LongType *tadOffsets, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  if (xType != yType) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execScalar requires both X & Y to have same data type";
    errorMessage += "X data type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += ", Y data type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);
    errorMessage += ", Z data type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());

  }
  if (zType != sd::DataType::BOOL) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execScalarBool requires Z to have bool data type";
    errorMessage += "X data type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += ", Y data type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);
    errorMessage += ", Z data type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  auto func = PRAGMA_THREADS_FOR {
    BUILD_DOUBLE_SELECTOR(
        xType, zType, functions::scalar::ScalarBoolTransform,
        ::transform(opNum, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo, hScalars, dimension, dimensionLength,
                    tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ, start, stop),
        SD_COMMON_TYPES_ALL, SD_BOOL_TYPES);
  };

  auto yLen = shape::length(hScalarShapeInfo);
  samediff::Threads::parallel_tad(func, 0, yLen, 1,
                                  sd::math::sd_min<int>(yLen, sd::Environment::getInstance().maxMasterThreads()));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarInt(sd::LaunchContext *lc, int opNum, const void *hX,
                                        const sd::LongType *hXShapeInfo, const void *dX,
                                        const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                        void *dZ, const sd::LongType *dZShapeInfo, const void *hScalar,
                                        const sd::LongType *hSscalarShapeInfo, const void *dScalar,
                                        const sd::LongType *dSscalarShapeInfo, void *extraParams,
                                        bool allowParallelism) {

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hSscalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  if (xType != yType || xType != zType) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execScalarInt requires both X & Y to have same data type";
    errorMessage += "X data type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += ", Y data type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);
    errorMessage += ", Z data type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
    
  }
  
  if (!sd::DataTypeUtils::isZ(zType)) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execScalarInt requires result type to be an integer type";
    errorMessage += "X data type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += ", Y data type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);
    errorMessage += ", Z data type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
    
  }
  
 
  auto func = PRAGMA_THREADS_FOR {
    BUILD_SINGLE_SELECTOR(xType, functions::scalar::ScalarIntTransform,
                          ::transform(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, hScalar, extraParams, start, stop),
                          SD_INTEGER_TYPES);
  };

  auto zLen = shape::length(hZShapeInfo);
  samediff::Threads::parallel_for(
      func, 0, zLen, 1,
      !allowParallelism
      ? 1
      : sd::math::sd_max<int>(
          1, sd::math::sd_min<int>(zLen / 1024, sd::Environment::getInstance().maxMasterThreads())));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarInt(
    sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo, const void *dX,
    const sd::LongType *dXShapeInfo, void *extraParams, void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
    const sd::LongType *dZShapeInfo, const void *hScalars, const sd::LongType *hScalarShapeInfo, const void *dScalars,
    const sd::LongType *dScalarShapeInfo,
    sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo,
    const sd::LongType *tadOffsets, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);



  if (xType != yType || xType != zType) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execScalarInt requires both X & Y to have same data type";
    errorMessage += "X data type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += ", Y data type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);
    errorMessage += ", Z data type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());

  }

  if (!sd::DataTypeUtils::isZ(zType)) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execScalarInt requires result type to be an integer type";
    errorMessage += "X data type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += ", Y data type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);
    errorMessage += ", Z data type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());

  }
  auto func = PRAGMA_THREADS_FOR {
    BUILD_SINGLE_SELECTOR(
        xType, functions::scalar::ScalarIntTransform,
        ::transform(opNum, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo, hScalars, dimension, dimensionLength,
                    tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ, start, stop),
        SD_INTEGER_TYPES);
  };

  auto yLen = shape::length(hScalarShapeInfo);
  samediff::Threads::parallel_tad(func, 0, yLen, 1,
                                  sd::math::sd_min<int>(yLen, sd::Environment::getInstance().maxMasterThreads()));
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @param hZ
 * @param hZShapeInfo
 */
void NativeOpExecutioner::execSummaryStats(sd::LaunchContext *lc, int opNum, const void *hX,
                                           const sd::LongType *hXShapeInfo, const void *dX,
                                           const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                           const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                           bool biasCorrected) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  BUILD_DOUBLE_SELECTOR(xType, zType, functions::summarystats::SummaryStatsReduce,
                        ::exec(opNum, biasCorrected, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo, nullptr, 1),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @param hZ
 * @param hZShapeInfo
 */
void NativeOpExecutioner::execSummaryStatsScalar(sd::LaunchContext *lc, int opNum, const void *hX,
                                                 const sd::LongType *hXShapeInfo, const void *dX,
                                                 const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                                 const sd::LongType *hZShapeInfo, void *dZ,
                                                 const sd::LongType *dZShapeInfo, bool biasCorrected) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  BUILD_DOUBLE_SELECTOR(xType, zType, functions::summarystats::SummaryStatsReduce,
                        ::execScalar(opNum, biasCorrected, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @param hZ
 * @param hZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void NativeOpExecutioner::execSummaryStats(sd::LaunchContext *lc, int opNum, const void *hX,
                                           const sd::LongType *hXShapeInfo, const void *dX,
                                           const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                           const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                           sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo,
                                           const sd::LongType *tadOffsets, bool biasCorrected) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::summarystats::SummaryStatsReduce,
      ::exec(opNum, biasCorrected, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo, dimension, dimensionLength),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param hX
 * @param xStride
 * @param hZ
 * @param resultStride
 * @param extraParams
 * @param n
 */
void NativeOpExecutioner::execTransformFloat(sd::LaunchContext *lc, int opNum, const void *hX,
                                             const sd::LongType *hXShapeInfo, const void *dX,
                                             const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                             void *dZ, const sd::LongType *dZShapeInfo, void *extraParams,
                                             const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  auto func = PRAGMA_THREADS_DO {
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformFloat,
                          ::exec(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, extraParams, thread_id, numThreads),
                          SD_COMMON_TYPES_ALL, SD_FLOAT_TYPES);
  };

  samediff::Threads::parallel_do(
      func, sd::math::sd_max<int>(1, sd::math::sd_min<int>(shape::length(hZShapeInfo) / 1024,
                                                           sd::Environment::getInstance().maxMasterThreads())));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformBool(sd::LaunchContext *lc, int opNum, const void *hX,
                                            const sd::LongType *hXShapeInfo, const void *dX,
                                            const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                            void *dZ, const sd::LongType *dZShapeInfo, void *extraParams,
                                            const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  auto func = PRAGMA_THREADS_DO {
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformBool,
                          ::exec(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, extraParams, thread_id, numThreads),
                          SD_COMMON_TYPES_ALL, SD_BOOL_TYPES);
  };

  samediff::Threads::parallel_do(
      func, sd::math::sd_max<int>(1, sd::math::sd_min<int>(shape::length(hZShapeInfo) / 1024,
                                                           sd::Environment::getInstance().maxMasterThreads())));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformAny(sd::LaunchContext *lc, int opNum, const void *hX,
                                           const sd::LongType *hXShapeInfo, const void *dX,
                                           const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                           void *dZ, const sd::LongType *dZShapeInfo, void *extraParams,
                                           const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets,
                                           bool allowParallelism) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);

  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  if(sd::DataTypeUtils::isS(xType)) {
    auto func = PRAGMA_THREADS_DO {
      BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformAny,
                            ::exec(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, extraParams, thread_id, numThreads),
                            SD_STRING_TYPES, SD_STRING_TYPES);
    };

    samediff::Threads::parallel_do(
        func, sd::math::sd_max<int>(1, sd::math::sd_min<int>(shape::length(hZShapeInfo) / 1024,
                                                             sd::Environment::getInstance().maxMasterThreads())));
  } else {
    auto func = PRAGMA_THREADS_DO {
      BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformAny,
                            ::exec(opNum,
                                   hX,
                                   hXShapeInfo,
                                   hZ, hZShapeInfo,
                                   extraParams,
                                   thread_id,
                                   numThreads),
                            SD_COMMON_TYPES, SD_COMMON_TYPES);
    };

    samediff::Threads::parallel_do(
        func, sd::math::sd_max<int>(1, sd::math::sd_min<int>(shape::length(hZShapeInfo) / 1024,
                                                             sd::Environment::getInstance().maxMasterThreads())));
  }


}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformSame(sd::LaunchContext *lc, int opNum, const void *hX,
                                            const sd::LongType *hXShapeInfo, const void *dX,
                                            const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                            void *dZ, const sd::LongType *dZShapeInfo, void *extraParams,
                                            const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);


  auto func = PRAGMA_THREADS_DO {
    BUILD_SINGLE_SELECTOR(xType, functions::transform::TransformSame,
                          ::exec(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, extraParams, thread_id, numThreads),
                          SD_COMMON_TYPES_ALL);
  };

  samediff::Threads::parallel_do(
      func, sd::math::sd_max<int>(1, sd::math::sd_min<int>(shape::length(hZShapeInfo) / 1024,
                                                           sd::Environment::getInstance().maxMasterThreads())));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformStrict(sd::LaunchContext *lc, int opNum, const void *hX,
                                              const sd::LongType *hXShapeInfo, const void *dX,
                                              const sd::LongType *dXShapeInfo, void *hZ,
                                              const sd::LongType *hZShapeInfo, void *dZ,
                                              const sd::LongType *dZShapeInfo, void *extraParams,
                                              const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  auto func = PRAGMA_THREADS_DO {
    BUILD_SINGLE_SELECTOR(xType, functions::transform::TransformStrict,
                          ::exec(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, extraParams, thread_id, numThreads),
                          SD_FLOAT_TYPES);
  };

  samediff::Threads::parallel_do(
      func, sd::math::sd_max<int>(1, sd::math::sd_min<int>(shape::length(hZShapeInfo) / 1024,
                                                           sd::Environment::getInstance().maxMasterThreads())));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(sd::LaunchContext *lc, int opNum, sd::Pointer state, void *hZ,
                                     const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                     void *extraArguments) {
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_SINGLE_SELECTOR(zType, functions::random::RandomFunction,
                        ::execTransform(opNum, state, hZ, hZShapeInfo, extraArguments), SD_FLOAT_TYPES);
  auto rng = reinterpret_cast<sd::graph::RandomGenerator *>(state);
  rng->rewindH(shape::length(hZShapeInfo));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(sd::LaunchContext *lc, int opNum, sd::Pointer state, const void *hX,
                                     const sd::LongType *hXShapeInfo, const void *dX, const sd::LongType *dXShapeInfo,
                                     void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
                                     const sd::LongType *dZShapeInfo, void *extraArguments) {
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  BUILD_SINGLE_SELECTOR(zType, functions::random::RandomFunction,
                        ::execTransform(opNum, state, hX, hXShapeInfo, hZ, hZShapeInfo, extraArguments),
                        SD_FLOAT_TYPES);

  auto rng = reinterpret_cast<sd::graph::RandomGenerator *>(state);
  rng->rewindH(shape::length(hZShapeInfo));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(sd::LaunchContext *lc, int opNum, sd::Pointer state, const void *hX,
                                     const sd::LongType *hXShapeInfo, const void *dX, const sd::LongType *dXShapeInfo,
                                     const void *hY, const sd::LongType *hYShapeInfo, const void *dY,
                                     const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                     void *dZ, const sd::LongType *dZShapeInfo, void *extraArguments) {
  auto xType = sd::ArrayOptions::dataType(hZShapeInfo);

  BUILD_SINGLE_SELECTOR(
      xType, functions::random::RandomFunction,
      ::execTransform(opNum, state, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, extraArguments), SD_FLOAT_TYPES);

  auto rng = reinterpret_cast<sd::graph::RandomGenerator *>(state);
  rng->rewindH(shape::length(hZShapeInfo));
}
