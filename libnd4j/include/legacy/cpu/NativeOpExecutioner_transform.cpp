/* ******************************************************************************
 *
 * Transform operations - uses SD_COMMON_TYPES, SD_FLOAT_TYPES, SD_BOOL_TYPES, SD_STRING_TYPES
 *
 ******************************************************************************/

// Selective rendering - MUST be included before types.h to define HAS_* flags
// Note: string_types.h removed to reduce file size - strings will be added separately later
#include <system/selective_rendering/core.h>
#include <system/selective_rendering/bool_types.h>
#include <system/selective_rendering/float_types.h>
#include <system/selective_rendering/bfloat_types.h>
#include <system/selective_rendering/int_types.h>
#include <system/selective_rendering/uint_types.h>

#include <array/DataTypeUtils.h>

#include <execution/Threads.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/transform_any.h>
#include <loops/transform_any_fp8.h>
#include <loops/transform_bool.h>
#include <loops/transform_float.h>
#include <loops/transform_same.h>
#include <loops/transform_strict.h>
#include <system/env_functions.h>
#include <types/types.h>

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformFloat(sd::LaunchContext *lc, int opNum, const void *hX,
                                             const sd::LongType *hXShapeInfo, const void *dX,
                                             const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                             void *dZ, const sd::LongType *dZShapeInfo, void *extraParams) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  auto func = PRAGMA_THREADS_DO {
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformFloat,
                          ::exec(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, extraParams, thread_id, numThreads),
                          SD_COMMON_TYPES, SD_FLOAT_TYPES);
  };

  samediff::Threads::parallel_do(
      func, sd::math::sd_max(1, sd::math::sd_min(shape::length(hZShapeInfo) / 1024,
                                                           sd::env_maxMasterThreads())));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformBool(sd::LaunchContext *lc, int opNum, const void *hX,
                                            const sd::LongType *hXShapeInfo, const void *dX,
                                            const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                            void *dZ, const sd::LongType *dZShapeInfo, void *extraParams) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  auto func = PRAGMA_THREADS_DO {
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformBool,
                          ::exec(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, extraParams, thread_id, numThreads),
                          SD_COMMON_TYPES, SD_BOOL_TYPES);
  };

  samediff::Threads::parallel_do(
      func, sd::math::sd_max(1, sd::math::sd_min(shape::length(hZShapeInfo) / 1024,
                                                           sd::env_maxMasterThreads())));
}

#if defined(HAS_FLOAT8)
////////////////////////////////////////////////////////////////////////
// Copies x of any storage type into the FP8 storage type F8 of z.
template <typename F8>
static void execTransformAnyToFp8(sd::DataType xType, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                  void *hZ, const sd::LongType *hZShapeInfo, void *extraParams, sd::LongType threadId,
                                  sd::LongType numThreads) {
  if (xType == sd::DataType::FLOAT8) {
    functions::transform::TransformAnyFp8<F8>::template execToFp8<sd::float8>(
        opNum, hX, hXShapeInfo, hZ, hZShapeInfo, extraParams, threadId, numThreads);
  } else if (xType == sd::DataType::FLOAT8_E5M2) {
    functions::transform::TransformAnyFp8<F8>::template execToFp8<sd::float8_e5m2>(
        opNum, hX, hXShapeInfo, hZ, hZShapeInfo, extraParams, threadId, numThreads);
  } else {
    BUILD_SINGLE_SELECTOR(xType, functions::transform::TransformAnyFp8<F8>::template execToFp8,
                          (opNum, hX, hXShapeInfo, hZ, hZShapeInfo, extraParams, threadId, numThreads),
                          SD_COMMON_TYPES);
  }
}

////////////////////////////////////////////////////////////////////////
// Copies x of the FP8 storage type F8 into z of a common type.
template <typename F8>
static void execTransformAnyFromFp8(sd::DataType zType, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                    void *hZ, const sd::LongType *hZShapeInfo, void *extraParams,
                                    sd::LongType threadId, sd::LongType numThreads) {
  BUILD_SINGLE_SELECTOR(zType, functions::transform::TransformAnyFp8<F8>::template execFromFp8,
                        (opNum, hX, hXShapeInfo, hZ, hZShapeInfo, extraParams, threadId, numThreads),
                        SD_COMMON_TYPES);
}
#endif

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformAny(sd::LaunchContext *lc, int opNum, const void *hX,
                                           const sd::LongType *hXShapeInfo, const void *dX,
                                           const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                           void *dZ, const sd::LongType *dZShapeInfo, void *extraParams,
                                           bool allowParallelism) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  // String type handling removed temporarily - will be added in separate file to manage compilation size
  {
    auto func = PRAGMA_THREADS_DO {
#if defined(HAS_FLOAT8)
      // FP8 is deliberately excluded from the arithmetic SD_COMMON_TYPES matrix.
      // A copy into or out of an FP8 storage type fixes that side and dispatches
      // the other; same-dtype copies stay storage copies.
      if (zType == sd::DataType::FLOAT8) {
        execTransformAnyToFp8<sd::float8>(xType, opNum, hX, hXShapeInfo, hZ, hZShapeInfo, extraParams, thread_id,
                                          numThreads);
        return;
      }
      if (zType == sd::DataType::FLOAT8_E5M2) {
        execTransformAnyToFp8<sd::float8_e5m2>(xType, opNum, hX, hXShapeInfo, hZ, hZShapeInfo, extraParams,
                                               thread_id, numThreads);
        return;
      }
      if (xType == sd::DataType::FLOAT8) {
        execTransformAnyFromFp8<sd::float8>(zType, opNum, hX, hXShapeInfo, hZ, hZShapeInfo, extraParams, thread_id,
                                            numThreads);
        return;
      }
      if (xType == sd::DataType::FLOAT8_E5M2) {
        execTransformAnyFromFp8<sd::float8_e5m2>(zType, opNum, hX, hXShapeInfo, hZ, hZShapeInfo, extraParams,
                                                 thread_id, numThreads);
        return;
      }
#endif
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
        func, sd::math::sd_max<sd::LongType,sd::LongType,sd::LongType>(1,
                                                                       sd::math::sd_min<sd::LongType,sd::LongType,sd::LongType>(shape::length(hZShapeInfo) / 1024,
                                                                                                                                sd::env_maxMasterThreads())));
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
                          SD_COMMON_TYPES);
  };

  samediff::Threads::parallel_do(
      func, sd::math::sd_max(1, sd::math::sd_min(shape::length(hZShapeInfo) / 1024,
                                                           sd::env_maxMasterThreads())));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformStrict(sd::LaunchContext *lc, int opNum, const void *hX,
                                              const sd::LongType *hXShapeInfo, const void *dX,
                                              const sd::LongType *dXShapeInfo, void *hZ,
                                              const sd::LongType *hZShapeInfo, void *dZ,
                                              const sd::LongType *dZShapeInfo, void *extraParams) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  auto func = PRAGMA_THREADS_DO {
    BUILD_SINGLE_SELECTOR(xType, functions::transform::TransformStrict,
                          ::exec(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, extraParams, thread_id, numThreads),
                          SD_FLOAT_TYPES);
  };

  samediff::Threads::parallel_do(
      func, sd::math::sd_max(1, sd::math::sd_min(shape::length(hZShapeInfo) / 1024,
                                                           sd::env_maxMasterThreads())));
}
