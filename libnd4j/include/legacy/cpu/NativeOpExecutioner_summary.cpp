/* ******************************************************************************
 *
 * SummaryStats operations - uses SD_COMMON_TYPES and SD_FLOAT_TYPES
 *
 ******************************************************************************/

// Selective rendering - MUST be included before types.h to define HAS_* flags
#include <system/selective_rendering/core.h>
#include <system/selective_rendering/bool_types.h>
#include <system/selective_rendering/float_types.h>
#include <system/selective_rendering/bfloat_types.h>
#include <system/selective_rendering/int_types.h>
#include <system/selective_rendering/uint_types.h>

#include <legacy/NativeOpExecutioner.h>
#include <loops/summarystatsreduce.h>
#include <types/types.h>

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execSummaryStats(sd::LaunchContext *lc, int opNum, const void *hX,
                                           sd::LongType *hXShapeInfo, const void *dX,
                                           sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                           sd::LongType *hZShapeInfo, void *dZ, sd::LongType *dZShapeInfo,
                                           bool biasCorrected) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::summarystats::SummaryStatsReduce,
                        ::exec(opNum, biasCorrected, const_cast<void *>(hX), hXShapeInfo, extraParams, hZ, hZShapeInfo, nullptr, 1),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execSummaryStatsScalar(sd::LaunchContext *lc, int opNum, const void *hX,
                                                 sd::LongType *hXShapeInfo, const void *dX,
                                                 sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                                 sd::LongType *hZShapeInfo, void *dZ,
                                                 sd::LongType *dZShapeInfo, bool biasCorrected) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::summarystats::SummaryStatsReduce,
                        ::execScalar(opNum, biasCorrected, const_cast<void *>(hX), hXShapeInfo, extraParams, hZ, hZShapeInfo),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execSummaryStats(sd::LaunchContext *lc, int opNum, const void *hX,
                                           sd::LongType *hXShapeInfo, const void *dX,
                                           sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                           sd::LongType *hZShapeInfo, void *dZ, sd::LongType *dZShapeInfo,
                                           sd::LongType *dimension, sd::LongType dimensionLength, sd::LongType *tadShapeInfo,
                                           sd::LongType *tadOffsets, bool biasCorrected) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::summarystats::SummaryStatsReduce,
      ::exec(opNum, biasCorrected, const_cast<void *>(hX), hXShapeInfo, extraParams, hZ, hZShapeInfo, dimension, dimensionLength),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);
}
