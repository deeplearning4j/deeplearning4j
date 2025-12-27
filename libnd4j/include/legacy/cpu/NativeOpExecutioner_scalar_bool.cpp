/* ******************************************************************************
 *
 * Scalar operations - BOOL TYPES ONLY (non-TAD version)
 *
 ******************************************************************************/

#include <array/DataTypeUtils.h>
#include <exceptions/datatype_exception.h>
#include <execution/Threads.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/scalar.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                         const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                                         const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                         const void *hScalar, const sd::LongType *hScalarShapeInfo, const void *dScalar,
                                         const sd::LongType *dScalarShapeInfo, void *extraParams, bool allowParallelism) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  // Only handle operations that result in boolean output
  if (zType != sd::DataType::BOOL) {
    return; // Let other files handle non-bool result types
  }

  auto func = PRAGMA_THREADS_FOR {
    BUILD_TRIPLE_SELECTOR(xType,yType,zType,functions::scalar::ScalarTransform,
                          ::transform(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, hScalar, extraParams, start, stop),
                          SD_COMMON_TYPES,SD_COMMON_TYPES,SD_BOOL_TYPES);
  };

  auto zLen = shape::length(hZShapeInfo);
  samediff::Threads::parallel_for(
      func, 0, zLen, 1,
      !allowParallelism
      ? 1
      : sd::math::sd_max<int>(
          1, sd::math::sd_min<int>(zLen / 1024, sd::Environment::getInstance().maxMasterThreads())));
}
