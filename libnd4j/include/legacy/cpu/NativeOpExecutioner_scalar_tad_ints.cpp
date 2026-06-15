/* ******************************************************************************
 *
 * Scalar TAD operations - INTEGER OUTPUT TYPES ONLY
 *
 ******************************************************************************/

#include <system/op_boilerplate.h>

#include <array/DataTypeUtils.h>
#include <exceptions/datatype_exception.h>
#include <execution/Threads.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/scalar.h>
#include <types/types.h>

////////////////////////////////////////////////////////////////////////
// TAD execScalarInt - integer output types only (13 x 13 x 9 = 1,521 combinations)
void NativeOpExecutioner::execScalarInt(sd::LaunchContext *lc, int opNum, void const *hX, sd::LongType const *hXShapeInfo,
                                        void const *dX, sd::LongType const *dXShapeInfo, void *extraParams, void *hZ,
                                        sd::LongType const *hZShapeInfo, void *dZ, sd::LongType const *dZShapeInfo,
                                        void const *hScalars, sd::LongType const *hScalarShapeInfo, void const *dScalars,
                                        sd::LongType const *dScalarShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength,
                                        sd::LongType const *tadShapeInfo, sd::LongType const *tadOffsets,
                                        sd::LongType const *tadShapeInfoZ, sd::LongType const *tadOffsetsZ) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  auto yLen = shape::length(hScalarShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    BUILD_TRIPLE_SELECTOR(xType, yType, zType, functions::scalar::ScalarTransform,
                          ::transform(opNum, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo, hScalars, dimension, dimensionLength,
                                    tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ, start, stop),
                          SD_COMMON_TYPES, SD_COMMON_TYPES, SD_INTEGER_TYPES);
  };

  samediff::Threads::parallel_tad(func, 0, yLen, 1, sd::math::sd_min<int>(yLen, sd::Environment::getInstance().maxMasterThreads()));
}
