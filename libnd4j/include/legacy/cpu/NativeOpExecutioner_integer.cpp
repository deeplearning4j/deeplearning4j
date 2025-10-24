/* ******************************************************************************
 *
 * Integer-only operations - uses ONLY integer types
 *
 ******************************************************************************/

// Selective rendering - MUST be included before types.h to define HAS_* flags
// Note: DataTypeUtils.h->logger.h uses SD_COMMON_TYPES_ALL, so we need all core types
#include <system/selective_rendering/core.h>
#include <system/selective_rendering/bool_types.h>
#include <system/selective_rendering/float_types.h>
#include <system/selective_rendering/bfloat_types.h>
#include <system/selective_rendering/int_types.h>
#include <system/selective_rendering/uint_types.h>

#include <array/DataTypeUtils.h>
#include <exceptions/datatype_exception.h>
#include <execution/Threads.h>
#include <helpers/LoopKind.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/broadcasting_int.h>
#include <loops/pairwise_int.h>
#include <loops/scalar_int.h>
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


  if (xType != yType || xType != zType)
    throw sd::datatype_exception::build("NativeOpExecutioner::execBroadcastInt", zType, xType, yType);

  if (!sd::DataTypeUtils::isZ(zType))
    throw sd::datatype_exception::build("NativeOpExecutioner::execBroadcastInt requires integer data type", zType);
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


  if (xType != yType || xType != zType)
    throw sd::datatype_exception::build("NativeOpExecutioner::execBroadcastInt", zType, xType, yType);

  if (!sd::DataTypeUtils::isZ(zType))
    throw sd::datatype_exception::build("NativeOpExecutioner::execBroadcastInt requires integer data type", zType);
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


  if (xType != yType || xType != zType)
    throw sd::datatype_exception::build("NativeOpExecutioner::execInverseBroadcastInt", zType, xType, yType);

  if (!sd::DataTypeUtils::isZ(zType))
    throw sd::datatype_exception::build("NativeOpExecutioner::execInverseBroadcastInt requires integer data type",
                                        zType);
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
void NativeOpExecutioner::execPairwiseIntTransform(sd::LaunchContext *lc, int opNum, const void *hX,
                                                   const sd::LongType *hXShapeInfo, const void *dX,
                                                   const sd::LongType *dXShapeInfo, const void *hY,
                                                   const sd::LongType *hYShapeInfo, const void *dY,
                                                   const sd::LongType *dYShapeInfo, void *hZ,
                                                   const sd::LongType *hZShapeInfo, void *dZ,
                                                   const sd::LongType *dZShapeInfo, void *extraParams) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  auto func = PRAGMA_THREADS_FOR {
    BUILD_SINGLE_SELECTOR(xType, functions::pairwise_transforms::PairWiseIntTransform,
                          ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, extraParams, start, stop),
                          SD_INTEGER_TYPES);
  };

  auto zLen = shape::length(hZShapeInfo);

  samediff::Threads::parallel_for(
      func, 0, zLen, 1,
      sd::math::sd_max<int>(1, sd::math::sd_min<int>(zLen / 1024, sd::Environment::getInstance().maxMasterThreads())));
}

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
      : sd::math::sd_max<int>(
          1, sd::math::sd_min<int>(zLen / 1024, sd::Environment::getInstance().maxMasterThreads())));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarInt(
    sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo, const void *dX,
    const sd::LongType *dXShapeInfo, void *extraParams, void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
    const sd::LongType *dZShapeInfo, const void *hScalars, const sd::LongType *hScalarShapeInfo, const void *dScalars,
    const sd::LongType *dScalarShapeInfo,
    sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo,
    const sd::LongType *tadOffsets, const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hScalarShapeInfo);
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
    BUILD_SINGLE_SELECTOR(
        xType, functions::scalar::ScalarIntTransform,
        ::transform(opNum, hX, hXShapeInfo, extraParams, hZ, hZShapeInfo, hScalars, dimension, dimensionLength,
                    tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ, start, stop),
        SD_INTEGER_TYPES);
  };

  auto yLen = shape::length(hScalarShapeInfo);
  samediff::Threads::parallel_tad(func, 0, yLen, 1,
                                  sd::math::sd_min<int>(yLen, sd::Environment::getInstance().maxMasterThreads()));
}
