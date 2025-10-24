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
#include <exceptions/datatype_exception.h>
#include <execution/Threads.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/transform_any.h>
#include <loops/transform_bool.h>
#include <loops/transform_float.h>
#include <loops/transform_same.h>
#include <loops/transform_strict.h>
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
                                            void *dZ, const sd::LongType *dZShapeInfo, void *extraParams) {
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
                                           bool allowParallelism) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  // String type handling removed temporarily - will be added in separate file to manage compilation size
  {
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
        func, sd::math::sd_max<sd::LongType,sd::LongType,sd::LongType>(1,
                                                                       sd::math::sd_min<sd::LongType,sd::LongType,sd::LongType>(shape::length(hZShapeInfo) / 1024,
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
                                              const sd::LongType *dZShapeInfo, void *extraParams) {
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
