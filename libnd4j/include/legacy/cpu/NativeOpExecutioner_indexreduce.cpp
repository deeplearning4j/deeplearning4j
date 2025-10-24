/* ******************************************************************************
 *
 * IndexReduce operations - uses SD_COMMON_TYPES and SD_INDEXING_TYPES
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
#include <loops/indexreduce.h>
#include <types/types.h>

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execIndexReduceScalar(sd::LaunchContext *lc, int opNum, const void *hX,
                                                const sd::LongType *hXShapeInfo, const void *dX,
                                                const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                                const sd::LongType *hZShapeInfo, void *dZ,
                                                const sd::LongType *dZShapeInfo) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  auto hz = reinterpret_cast<sd::LongType *>(hZ);
  BUILD_DOUBLE_SELECTOR(xType, zType, hz[0] = functions::indexreduce::IndexReduce,
                        ::execScalar(opNum, hX, hXShapeInfo, extraParams), SD_COMMON_TYPES, SD_INDEXING_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execIndexReduce(sd::LaunchContext *lc, int opNum, const void *hX,
                                          const sd::LongType *hXShapeInfo, const void *dX,
                                          const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                          const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                          sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo,
                                          const sd::LongType *tadOffsets) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  auto hz = reinterpret_cast<sd::LongType *>(hZ);
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::indexreduce::IndexReduce,
                        ::exec(opNum, hX, hXShapeInfo, extraParams, hz, hZShapeInfo, dimension, dimensionLength,
                               tadShapeInfo, tadOffsets),
                        SD_COMMON_TYPES, SD_INDEXING_TYPES);
}
