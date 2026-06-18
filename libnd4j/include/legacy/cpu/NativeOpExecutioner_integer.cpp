/* ******************************************************************************
 *
 * Integer-only operations - uses ONLY integer types
 *
 ******************************************************************************/

// Selective rendering - MUST be included before types.h to define HAS_* flags
// Note: DataTypeUtils.h->logger.h uses SD_COMMON_TYPES, so we need all core types
#include <system/selective_rendering/core.h>
#include <system/selective_rendering/bool_types.h>
#include <system/selective_rendering/float_types.h>
#include <system/selective_rendering/bfloat_types.h>
#include <system/selective_rendering/int_types.h>
#include <system/selective_rendering/uint_types.h>

#include <array/DataTypeUtils.h>
#include <execution/Threads.h>
#include <helpers/LoopKind.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/broadcasting_int.h>
#include <loops/pairwise_int.h>
#include <loops/scalar_int.h>
#include <system/env_functions.h>
#include <types/types.h>

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcastInt(
    sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo, const void *dX,
    const sd::LongType *dXShapeInfo, const void *hY, const sd::LongType *hYShapeInfo, const void *dY,
    const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
    const sd::LongType *dZShapeInfo,sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo,
    const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);


  if (xType != yType || xType != zType) {
    std::string dtMsg = std::string("NativeOpExecutioner::execBroadcastInt") + "; Expected: [" + sd::DataTypeUtils::asString(zType) + "]; Actual: [" + sd::DataTypeUtils::asString(xType) + ", " + sd::DataTypeUtils::asString(yType) + "]";
    THROW_EXCEPTION(dtMsg.c_str());
  }

  if (!sd::DataTypeUtils::isZ(zType)) {
    std::string dtMsg = std::string("NativeOpExecutioner::execBroadcastInt requires integer data type") + "; Actual: [" + sd::DataTypeUtils::asString(zType) + "]";
    THROW_EXCEPTION(dtMsg.c_str());
  }
  auto func = PRAGMA_THREADS_FOR {
    BUILD_SINGLE_SELECTOR(xType, functions::broadcast::BroadcastInt,
                          ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, dimension, dimensionLength,
                                 tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ, start, stop),
                          SD_INTEGER_TYPES);
  };

  auto xLen = shape::length(hXShapeInfo);
  auto yLen = shape::length(hYShapeInfo);
  auto numTads = xLen / yLen;

  samediff::Threads::parallel_tad(func, 0, numTads);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcastInt(sd::LaunchContext *lc, const int opNum, const void *hX,
                                           const sd::LongType *hXShapeInfo, const void *dX,
                                           const sd::LongType *dXShapeInfo, const void *hY,
                                           const sd::LongType *hYShapeInfo, const void *dY,
                                           const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                           void *dZ, const sd::LongType *dZShapeInfo) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);


  if (xType != yType || xType != zType) {
    std::string dtMsg = std::string("NativeOpExecutioner::execBroadcastInt") + "; Expected: [" + sd::DataTypeUtils::asString(zType) + "]; Actual: [" + sd::DataTypeUtils::asString(xType) + ", " + sd::DataTypeUtils::asString(yType) + "]";
    THROW_EXCEPTION(dtMsg.c_str());
  }

  if (!sd::DataTypeUtils::isZ(zType)) {
    std::string dtMsg = std::string("NativeOpExecutioner::execBroadcastInt requires integer data type") + "; Actual: [" + sd::DataTypeUtils::asString(zType) + "]";
    THROW_EXCEPTION(dtMsg.c_str());
  }
  BUILD_SINGLE_SELECTOR(xType, functions::broadcast::BroadcastInt,
                        ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo), SD_INTEGER_TYPES);
}

void NativeOpExecutioner::execInverseBroadcastInt(
    sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo, const void *dX,
    const sd::LongType *dXShapeInfo, const void *hY, const sd::LongType *hYShapeInfo, const void *dY,
    const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
    const sd::LongType *dZShapeInfo,sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo,
    const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);


  if (xType != yType || xType != zType) {
    std::string dtMsg = std::string("NativeOpExecutioner::execInverseBroadcastInt") + "; Expected: [" + sd::DataTypeUtils::asString(zType) + "]; Actual: [" + sd::DataTypeUtils::asString(xType) + ", " + sd::DataTypeUtils::asString(yType) + "]";
    THROW_EXCEPTION(dtMsg.c_str());
  }

  if (!sd::DataTypeUtils::isZ(zType)) {
    std::string dtMsg = std::string("NativeOpExecutioner::execInverseBroadcastInt requires integer data type") + "; Actual: [" + sd::DataTypeUtils::asString(zType) + "]";
    THROW_EXCEPTION(dtMsg.c_str());
  }
  auto func = PRAGMA_THREADS_FOR {
    BUILD_SINGLE_SELECTOR(
        xType, functions::broadcast::BroadcastInt,
        ::execInverse(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, dimension, dimensionLength,
                      tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ, start, stop),
        SD_INTEGER_TYPES);
  };

  auto xLen = shape::length(hXShapeInfo);
  auto yLen = shape::length(hYShapeInfo);
  auto numTads = yLen / xLen;

  samediff::Threads::parallel_tad(func, 0, numTads);
}

////////////////////////////////////////////////////////////////////////
// execPairwiseIntTransform moved to NativeOpExecutioner_pairwise_int.cpp
////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarInt(sd::LaunchContext *lc, int opNum, const void *hX,
                                        const sd::LongType *hXShapeInfo, const void *dX,
                                        const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                        void *dZ, const sd::LongType *dZShapeInfo, const void *hScalar,
                                        const sd::LongType *hSscalarShapeInfo, const void *dScalar,
                                        const sd::LongType *dSscalarShapeInfo, void *extraParams,
                                        bool allowParallelism) {

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hSscalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (!sd::DataTypeUtils::isZ(zType)) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execScalarInt requires result type to be an integer type";
    errorMessage += "X data type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += ", Y data type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);
    errorMessage += ", Z data type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());

  }

  auto func = PRAGMA_THREADS_FOR {
    BUILD_SINGLE_SELECTOR(xType, functions::scalar::ScalarIntTransform,
                          ::transform(opNum, hX, hXShapeInfo, hZ, hZShapeInfo, hScalar, extraParams, start, stop),
                          SD_INTEGER_TYPES);
  };

  auto zLen = shape::length(hZShapeInfo);
  samediff::Threads::parallel_for(
      func, 0, zLen, 1,
      !allowParallelism
      ? 1
      : sd::math::sd_max(
          1, sd::math::sd_min(zLen / 1024, sd::env_maxMasterThreads())));
}

////////////////////////////////////////////////////////////////////////
// TAD execScalarInt moved to NativeOpExecutioner_scalar_tad_ints.cpp
