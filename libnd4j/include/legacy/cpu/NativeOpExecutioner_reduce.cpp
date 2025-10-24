/* ******************************************************************************
 *
 * Reduce operations - uses SD_NUMERIC_TYPES, SD_FLOAT_TYPES, SD_COMMON_TYPES, SD_BOOL_TYPES
 *
 ******************************************************************************/

// Selective rendering - MUST be included before types.h to define HAS_* flags
#include <system/selective_rendering/core.h>
#include <system/selective_rendering/bool_types.h>
#include <system/selective_rendering/float_types.h>
#include <system/selective_rendering/bfloat_types.h>
#include <system/selective_rendering/int_types.h>
#include <system/selective_rendering/uint_types.h>

#include <array/TadPack.h>
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/reduce3.h>
#include <loops/reduce_bool.h>
#include <loops/reduce_float.h>
#include <loops/reduce_long.h>
#include <loops/reduce_same.h>
#include <types/types.h>

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceFloat(sd::LaunchContext *lc, int opNum, const void *hX,
                                          const sd::LongType *hXShapeInfo, const void *dX,
                                          const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                          const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                          sd::LongType *dimension, sd::LongType dimensionLength) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::reduce::ReduceFloatFunction,
      ::exec(opNum, lc ? lc->getWorkspace() : nullptr, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo, dimension),
      SD_NUMERIC_TYPES, SD_FLOAT_TYPES);
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
        SD_NUMERIC_TYPES);
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
      SD_NUMERIC_TYPES, SD_LONG_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceFloatScalar(sd::LaunchContext *lc, int opNum, const void *hX,
                                                const sd::LongType *hXShapeInfo, const void *dX,
                                                const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                                const sd::LongType *hZShapeInfo, void *dZ,
                                                const sd::LongType *dZShapeInfo) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceFloatFunction,
                        ::execScalar(opNum, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo), SD_NUMERIC_TYPES,
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
                        ::execScalar(opNum, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo), SD_NUMERIC_TYPES);
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
void NativeOpExecutioner::execReduce3(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                      const void *dX, const sd::LongType *dXShapeInfo, void *extraParamsVals,
                                      const void *hY, const sd::LongType *hYShapeInfo, const void *dY,
                                      const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                      void *dZ, const sd::LongType *dZShapeInfo) {
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
  sd::LongType *castedConst = const_cast<sd::LongType *>(hXShapeInfo);
  sd::LongType *hYShapeInfoNonConst = const_cast<sd::LongType *>(hYShapeInfo);
  if (xLen == yLen) {
    tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(castedConst, dimension, dimensionLength);
  } else if (yLen > xLen) {
    tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(hYShapeInfoNonConst, dimension, dimensionLength);
  } else {
    tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(castedConst, dimension, dimensionLength);
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

  auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(const_cast<sd::LongType *>(hXShapeInfo), dimension, dimensionLength);
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
  sd::LongType *castedConst = const_cast<sd::LongType *>(hXShapeInfo);
  sd::LongType *hYShapeInfoNonConst = const_cast<sd::LongType *>(hYShapeInfo);
  if (xLen == yLen) {
    tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(castedConst, dimension, dimensionLength);
  } else if (yLen > xLen) {
    tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(hYShapeInfoNonConst, dimension, dimensionLength);
  } else {
    tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(castedConst, dimension, dimensionLength);
  }
  auto func = PRAGMA_THREADS_FOR {
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3,
                          ::exec(opNum, hX, hXShapeInfo, extraParamsVals, hY, hYShapeInfo, hZ, hZShapeInfo, dimension,
                                 dimensionLength, tadShapeInfo, tadOffsets, start, stop),
                          SD_NUMERIC_TYPES, SD_FLOAT_TYPES);
  };

  samediff::Threads::parallel_tad(func, 0, tadPack->numberOfTads());
}
