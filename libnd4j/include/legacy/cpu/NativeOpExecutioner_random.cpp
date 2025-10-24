/* ******************************************************************************
 *
 * Random operations - uses ONLY float types
 *
 ******************************************************************************/

// Selective rendering - MUST be included before types.h to define HAS_* flags
// Note: NativeOpExecutioner.h->NDArray.h uses SD_COMMON_TYPES, so we need all core types
#include <system/selective_rendering/core.h>
#include <system/selective_rendering/bool_types.h>
#include <system/selective_rendering/float_types.h>
#include <system/selective_rendering/bfloat_types.h>
#include <system/selective_rendering/int_types.h>
#include <system/selective_rendering/uint_types.h>

#include <legacy/NativeOpExecutioner.h>
#include <loops/random.h>
#include <types/types.h>

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(sd::LaunchContext *lc, int opNum, sd::Pointer state, void *hZ,
                                     const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                     void *extraArguments) {
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_SINGLE_SELECTOR(zType, functions::random::RandomFunction,
                        ::execTransform(opNum, state, hZ, hZShapeInfo, extraArguments), SD_FLOAT_TYPES);
  auto rng = reinterpret_cast<sd::graph::RandomGenerator *>(state);
  rng->rewindH(shape::length(hZShapeInfo));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(sd::LaunchContext *lc, int opNum, sd::Pointer state, const void *hX,
                                     const sd::LongType *hXShapeInfo, const void *dX, const sd::LongType *dXShapeInfo,
                                     void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
                                     const sd::LongType *dZShapeInfo, void *extraArguments) {
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_SINGLE_SELECTOR(zType, functions::random::RandomFunction,
                        ::execTransform(opNum, state, hX, hXShapeInfo, hZ, hZShapeInfo, extraArguments),
                        SD_FLOAT_TYPES);
  auto rng = reinterpret_cast<sd::graph::RandomGenerator *>(state);
  rng->rewindH(shape::length(hZShapeInfo));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(sd::LaunchContext *lc, int opNum, sd::Pointer state, const void *hX,
                                     const sd::LongType *hXShapeInfo, const void *dX, const sd::LongType *dXShapeInfo,
                                     const void *hY, const sd::LongType *hYShapeInfo, const void *dY,
                                     const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                     void *dZ, const sd::LongType *dZShapeInfo, void *extraArguments) {
  auto xType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_SINGLE_SELECTOR(
      xType, functions::random::RandomFunction,
      ::execTransform(opNum, state, hX, hXShapeInfo, hY, hYShapeInfo, hZ, hZShapeInfo, extraArguments), SD_FLOAT_TYPES);
  auto rng = reinterpret_cast<sd::graph::RandomGenerator *>(state);
  rng->rewindH(shape::length(hZShapeInfo));
}
