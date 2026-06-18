/* ******************************************************************************
 *
 * Scalar TAD operations - BOOL TYPES ONLY
 * Uses ScalarBoolTransform with BUILD_DOUBLE_SELECTOR (X input type, Z=bool output)
 *
 ******************************************************************************/

#include <array/DataTypeUtils.h>

#include <execution/Threads.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/scalar_bool.h>
#include <system/env_functions.h>
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
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  // Only handle operations that result in boolean output
  if (zType != sd::DataType::BOOL) {
    return; // Let other files handle non-bool result types
  }

  auto func = PRAGMA_THREADS_FOR {
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::scalar::ScalarBoolTransform,
                          ::transform(opNum, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo, hScalars, dimension, dimensionLength,
                                     tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ, start, stop),
                          SD_COMMON_TYPES, SD_BOOL_TYPES);
  };

  auto yLen = shape::length(hScalarShapeInfo);
  samediff::Threads::parallel_tad(func, 0, yLen, 1,
                                  sd::math::sd_min(yLen, sd::env_maxMasterThreads()));
}
