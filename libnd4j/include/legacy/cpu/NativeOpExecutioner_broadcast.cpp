/* ******************************************************************************
 *
 * Broadcast operations (non-integer) - uses SD_COMMON_TYPES
 *
 ******************************************************************************/

// Selective rendering - MUST be included before types.h to define HAS_* flags
#include <system/selective_rendering/core.h>
#include <system/selective_rendering/bool_types.h>
#include <system/selective_rendering/float_types.h>
#include <system/selective_rendering/bfloat_types.h>
#include <system/selective_rendering/int_types.h>
#include <system/selective_rendering/uint_types.h>

#include <exceptions/datatype_exception.h>
#include <execution/Threads.h>
#include <helpers/LoopKind.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/broadcasting.h>
#include <loops/broadcasting_bool.h>
#include <types/types.h>

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcast(sd::LaunchContext *lc, int opNum, const void *hX,
                                        const sd::LongType *hXShapeInfo, const void *dX,
                                        const sd::LongType *dXShapeInfo, const void *hY,
                                        const sd::LongType *hYShapeInfo, const void *dY,
                                        const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                        void *dZ, const sd::LongType *dZShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength,
                                        const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets,
                                        const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  auto loopKind = sd::LoopKind::deduceKindOfLoopBroadcast(hXShapeInfo, hYShapeInfo, hZShapeInfo);
  auto func = PRAGMA_THREADS_FOR {
    BUILD_SINGLE_SELECTOR_THRICE(
        xType, functions::broadcast::Broadcast,
        ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo,
               tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ, loopKind, start, stop),
        SD_COMMON_TYPES);
  };

  sd::LongType numTads = shape::tensorsAlongDimension(hXShapeInfo, dimension, dimensionLength);
  samediff::Threads::parallel_tad(func, 0, numTads);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcast(sd::LaunchContext *lc, const int opNum, const void *hX,
                                        const sd::LongType *hXShapeInfo, const void *dX,
                                        const sd::LongType *dXShapeInfo, const void *hY,
                                        const sd::LongType *hYShapeInfo, const void *dY,
                                        const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                        void *dZ, const sd::LongType *dZShapeInfo) {

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  BUILD_SINGLE_SELECTOR_THRICE(xType, functions::broadcast::Broadcast,
                               ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo), SD_COMMON_TYPES);
}

void NativeOpExecutioner::execInverseBroadcast(
    sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo, const void *dX,
    const sd::LongType *dXShapeInfo, const void *hY, const sd::LongType *hYShapeInfo, const void *dY,
    const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
    const sd::LongType *dZShapeInfo,sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo,
    const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);


  if (!sd::Environment::getInstance().isExperimentalBuild())
    if ((yType != xType && yType != sd::DataType::BOOL) || xType != zType)
      throw sd::datatype_exception::build("NativeOps::execBroadcast both operands must have same data type", xType,
                                          yType);

  auto func = PRAGMA_THREADS_FOR {
    BUILD_SINGLE_SELECTOR_THRICE(
        xType, functions::broadcast::Broadcast,
        ::execInverse(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, dimension, dimensionLength,
                      tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ, start, stop),
        SD_COMMON_TYPES);
  };

  auto xLen = shape::length(hXShapeInfo);
  auto yLen = shape::length(hYShapeInfo);
  auto numTads = yLen / xLen;

  samediff::Threads::parallel_tad(func, 0, numTads);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcastBool(sd::LaunchContext *lc, int opNum, const void *hX,
                                            const sd::LongType *hXShapeInfo, const void *dX,
                                            const sd::LongType *dXShapeInfo, const void *hY,
                                            const sd::LongType *hYShapeInfo, const void *dY,
                                            const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                            void *dZ, const sd::LongType *dZShapeInfo, void *extraParams,
                                            sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo,
                                            const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ,
                                            const sd::LongType *tadOffsetsZ) {

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  auto func = PRAGMA_THREADS_FOR {
    BUILD_DOUBLE_SELECTOR(
        xType, zType, functions::broadcast::BroadcastBool,
        ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, extraParams, dimension, dimensionLength,
               tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ, start, stop),
        SD_COMMON_TYPES, SD_BOOL_TYPES);
  };
  auto xLen = shape::length(hXShapeInfo);
  auto yLen = shape::length(hYShapeInfo);
  auto numTads = xLen / yLen;

  samediff::Threads::parallel_tad(func, 0, numTads);

}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcastBool(sd::LaunchContext *lc, const int opNum, const void *hX,
                                            const sd::LongType *hXShapeInfo, const void *dX,
                                            const sd::LongType *dXShapeInfo, const void *hY,
                                            const sd::LongType *hYShapeInfo, const void *dY,
                                            const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                            void *dZ, const sd::LongType *dZShapeInfo, void *extraParams) {

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::broadcast::BroadcastBool,
                        ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, extraParams), SD_COMMON_TYPES,
                        SD_BOOL_TYPES);
}

void NativeOpExecutioner::execInverseBroadcastBool(
    sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo, const void *dX,
    const sd::LongType *dXShapeInfo, const void *hY, const sd::LongType *hYShapeInfo, const void *dY,
    const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
    const sd::LongType *dZShapeInfo, void *extraParams,sd::LongType *dimension, sd::LongType dimensionLength,
    const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ,
    const sd::LongType *tadOffsetsZ) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);


  if (!sd::Environment::getInstance().isExperimentalBuild())
    if (yType != xType || sd::DataType::BOOL != zType)
      throw sd::datatype_exception::build("NativeOps::execInverseBroadcastBool both operands must have same data type",
                                          xType, yType);

  auto func = PRAGMA_THREADS_FOR {
    BUILD_DOUBLE_SELECTOR(
        xType, zType, functions::broadcast::BroadcastBool,
        ::execInverse(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, extraParams, dimension, dimensionLength,
                      tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ, start, stop),
        SD_COMMON_TYPES, SD_BOOL_TYPES);
  };
  auto xLen = shape::length(hXShapeInfo);
  auto yLen = shape::length(hYShapeInfo);
  auto numTads = yLen / xLen;

  samediff::Threads::parallel_tad(func, 0, numTads);

}
