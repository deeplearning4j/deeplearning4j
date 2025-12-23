/* ******************************************************************************
 *
 * Scalar TAD operations - BOOL TYPES ONLY
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
// TAD version of execScalarBool for bool output types
// NOTE: extraParams comes BEFORE hZ in the TAD version signature!
void NativeOpExecutioner::execScalarBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                         const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                         const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                         const void *hScalars, const sd::LongType *hScalarShapeInfo, const void *dScalars,
                                         const sd::LongType *dScalarShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength,
                                         const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets,
                                         const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  // Only handle operations that result in boolean output
  if (zType != sd::DataType::BOOL) {
    return; // Let other files handle non-bool result types
  }

  auto func = PRAGMA_THREADS_FOR {
    BUILD_TRIPLE_SELECTOR(xType,yType,zType,functions::scalar::ScalarTransform,
                          ::transform(opNum, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo, hScalars, dimension, dimensionLength,
                                     tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ, start, stop),
                          SD_COMMON_TYPES,SD_COMMON_TYPES,SD_BOOL_TYPES);
  };

  auto yLen = shape::length(hScalarShapeInfo);
  samediff::Threads::parallel_tad(func, 0, yLen, 1,
                                  sd::math::sd_min<int>(yLen, sd::Environment::getInstance().maxMasterThreads()));
}
