/* ******************************************************************************
 *
 * Scalar operations - BOOL TYPES ONLY (non-TAD version)
 * Uses ScalarBoolTransform with BUILD_DOUBLE_SELECTOR (X input type, Z=bool output)
 *
 ******************************************************************************/

#include <array/DataTypeUtils.h>
#include <exceptions/datatype_exception.h>
#include <execution/Threads.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/scalar_bool.h>
#include <system/env_functions.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                         const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                                         const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                         const void *hScalar, const sd::LongType *hScalarShapeInfo, const void *dScalar,
                                         const sd::LongType *dScalarShapeInfo, void *extraParams, bool allowParallelism) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  // Only handle operations that result in boolean output
  if (zType != sd::DataType::BOOL) {
    return; // Let other files handle non-bool result types
  }

  auto func = PRAGMA_THREADS_FOR {
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::scalar::ScalarBoolTransform,
                          ::transform(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, hScalar, extraParams, start, stop),
                          SD_COMMON_TYPES, SD_BOOL_TYPES);
  };

  auto zLen = shape::length(hZShapeInfo);
  samediff::Threads::parallel_for(
      func, 0, zLen, 1,
      !allowParallelism
      ? 1
      : sd::math::sd_max(
          1, sd::math::sd_min(zLen / 1024, sd::env_maxMasterThreads())));
}
