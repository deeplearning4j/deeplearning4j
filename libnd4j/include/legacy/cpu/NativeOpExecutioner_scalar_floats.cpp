/* ******************************************************************************
 *
 * Scalar operations - FLOAT TYPES ONLY
 *
 ******************************************************************************/

// Selective rendering - include ALL types (this is the master implementation)
#include <system/op_boilerplate.h>

#include <array/DataTypeUtils.h>
#include <exceptions/datatype_exception.h>
#include <execution/Threads.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/scalar.h>
#include <types/types.h>

////////////////////////////////////////////////////////////////////////
// Non-TAD execScalar using BUILD_TRIPLE_SELECTOR (manageable with 4 types = 64 combinations)
void NativeOpExecutioner::execScalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                     const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                                     const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                     const void *hScalar, const sd::LongType *hScalarShapeInfo, const void *dScalar,
                                     const sd::LongType *dScalarShapeInfo, void *extraParams, bool allowParallelism) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  auto zLen = shape::length(hZShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    BUILD_TRIPLE_SELECTOR(xType, yType, zType, functions::scalar::ScalarTransform,
                          ::transform(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, hScalar, extraParams, start, stop),
                          SD_COMMON_TYPES, SD_COMMON_TYPES, SD_COMMON_TYPES);
  };

  samediff::Threads::parallel_for(
      func, 0, zLen, 1,
      !allowParallelism ? 1 : sd::math::sd_max<int>(1, sd::math::sd_min<int>(zLen / 1024, sd::Environment::getInstance().maxMasterThreads())));
}
