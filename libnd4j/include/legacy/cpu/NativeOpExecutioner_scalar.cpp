/* ******************************************************************************
 *
 * Scalar operations (non-integer) - uses SD_NUMERIC_TYPES, SD_COMMON_TYPES, SD_BOOL_TYPES
 *
 ******************************************************************************/

// Selective rendering - MUST be included before types.h to define HAS_* flags
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
#include <loops/scalar.h>
#include <types/types.h>

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                     const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                                     const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                     const void *hScalar, const sd::LongType *hScalarShapeInfo, const void *dScalar,
                                     const sd::LongType *dScalarShapeInfo, void *extraParams, bool allowParallelism) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  auto func = PRAGMA_THREADS_FOR {
    BUILD_TRIPLE_SELECTOR(xType,yType,zType,functions::scalar::ScalarTransform,
                          ::transform(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, hScalar, extraParams, start, stop),
                          SD_NUMERIC_TYPES,SD_NUMERIC_TYPES,SD_NUMERIC_TYPES
                          );
  };

  auto zLen = shape::length(hZShapeInfo);
  samediff::Threads::parallel_for(
      func, 0, zLen, 1,
      !allowParallelism
      ? 1
      : sd::math::sd_max<int>(
          1, sd::math::sd_min<int>(zLen / 1024, sd::Environment::getInstance().maxMasterThreads())));
}

// execScalar TAD function moved to NativeOpExecutioner_scalar_tad.cpp to reduce compilation size
// execScalarBool functions moved to NativeOpExecutioner_scalar_bool.cpp to reduce compilation size
