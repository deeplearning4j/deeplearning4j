/* ******************************************************************************
 *
 * Pairwise operations (non-integer) - uses SD_NUMERIC_TYPES
 *
 ******************************************************************************/

// Selective rendering - MUST be included before types.h to define HAS_* flags
#include <system/selective_rendering/core.h>
#include <system/selective_rendering/bool_types.h>
#include <system/selective_rendering/float_types.h>
#include <system/selective_rendering/bfloat_types.h>
#include <system/selective_rendering/int_types.h>
#include <system/selective_rendering/uint_types.h>

#include <array/DataTypeUtils.h>
#include <exceptions/datatype_exception.h>
#include <execution/Threads.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/pairwise_bool.h>
#include <loops/pairwise_transform.h>
#include <types/types.h>

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execPairwiseTransform(sd::LaunchContext *lc, int opNum,
                                                const void *hX,
                                                const sd::LongType *hXShapeInfo,
                                                const void *dX,
                                                const sd::LongType *dXShapeInfo,
                                                const void *hY,
                                                const sd::LongType *hYShapeInfo,
                                                const void *dY,
                                                const sd::LongType *dYShapeInfo,
                                                void *hZ,
                                                const sd::LongType *hZShapeInfo,
                                                void *dZ,
                                                const sd::LongType *dZShapeInfo,
                                                void *extraParams) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  auto func = PRAGMA_THREADS_FOR {
    BUILD_TRIPLE_SELECTOR(xType, yType, zType, functions::pairwise_transforms::PairWiseTransform,
                          ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, extraParams, start, stop),
                          SD_NUMERIC_TYPES, SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);
  };

  auto zLen = shape::length(hZShapeInfo);
  samediff::Threads::parallel_for(
      func, 0, zLen, 1,
      sd::math::sd_max<int>(1, sd::math::sd_min<int>(zLen / 1024, sd::Environment::getInstance().maxMasterThreads())));



}


////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execPairwiseBoolTransform(sd::LaunchContext *lc, int opNum, const void *hX,
                                                    const sd::LongType *hXShapeInfo, const void *dX,
                                                    const sd::LongType *dXShapeInfo, const void *hY,
                                                    const sd::LongType *hYShapeInfo, const void *dY,
                                                    const sd::LongType *dYShapeInfo, void *hZ,
                                                    const sd::LongType *hZShapeInfo, void *dZ,
                                                    const sd::LongType *dZShapeInfo, void *extraParams) {
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  if (zType != sd::DataType::BOOL) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execPairwiseBoolTransform";
    errorMessage += " zType must be BOOL";
    errorMessage += " zType: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  auto func = PRAGMA_THREADS_FOR {
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::pairwise_transforms::PairWiseBoolTransform,
                          ::exec(opNum, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, extraParams, start, stop),
                          SD_COMMON_TYPES, SD_BOOL_TYPES);
  };

  auto zLen = shape::length(hZShapeInfo);
  samediff::Threads::parallel_for(
      func, 0, zLen, 1,
      sd::math::sd_max<int>(1, sd::math::sd_min<int>(zLen / 1024, sd::Environment::getInstance().maxMasterThreads())));
}
