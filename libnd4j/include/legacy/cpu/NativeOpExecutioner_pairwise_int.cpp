/* ******************************************************************************
 *
 * Pairwise operations - INTEGER TYPES OUTPUT ONLY
 *
 ******************************************************************************/

#include <system/op_boilerplate.h>

#include <array/DataTypeUtils.h>
#include <exceptions/datatype_exception.h>
#include <execution/Threads.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/pairwise_int.h>
#include <types/types.h>

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execPairwiseIntTransform(sd::LaunchContext *lc, int opNum, const void *hX,
                                                   const sd::LongType *hXShapeInfo, const void *dX,
                                                   const sd::LongType *dXShapeInfo, const void *hY,
                                                   const sd::LongType *hYShapeInfo, const void *dY,
                                                   const sd::LongType *dYShapeInfo, void *hZ,
                                                   const sd::LongType *hZShapeInfo, void *dZ,
                                                   const sd::LongType *dZShapeInfo, void *extraParams) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    // PairWiseIntTransform only takes ONE template parameter (X type)
    BUILD_SINGLE_SELECTOR(xType, functions::pairwise_transforms::PairWiseIntTransform,
                          ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, extraParams, start, stop),
                          SD_COMMON_TYPES);
  };

  auto zLen = shape::length(hZShapeInfo);
  samediff::Threads::parallel_for(
      func, 0, zLen, 1,
      sd::math::sd_max<int>(1, sd::math::sd_min<int>(zLen / 1024, sd::Environment::getInstance().maxMasterThreads())));
}
