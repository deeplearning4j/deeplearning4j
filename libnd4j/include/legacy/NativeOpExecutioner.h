/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by agibsonccc on 1/28/16.
//

#ifndef NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H
#define NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H
#pragma  once
#include <array/NDArray.h>  // Ensure this is included first

#include <array/ArrayOptions.hXX>
#include <execution/LaunchContext.h>
#include <ops/specials.h>
#include <ops/specials_sparse.h>
#include <types/types.h>
#include <helpers/shape.h>
#include <cstddef>
#include <limits>

#ifndef __JAVACPP_HACK__
namespace sd {
/** Borrowed operand metadata. Offsets are relative to the original array, in elements. */
struct SD_LIB_EXPORT LegacyTensorArg {
  const NDArray* array;
  const LongType* hostShapeInfo;
  const LongType* deviceShapeInfo;
  LongType relativeElementOffset;

  static LegacyTensorArg fromArray(const NDArray* array) {
    if (array == nullptr) THROW_EXCEPTION("LegacyTensorArg requires an NDArray");
    auto* borrowed = const_cast<NDArray*>(array);
    return withShape(array, borrowed->shapeInfo(), borrowed->specialShapeInfo());
  }

  static LegacyTensorArg withShape(const NDArray* array, const LongType* hostShape,
                                   const LongType* deviceShape,
                                   LongType relativeElementOffset = 0) {
    if (array == nullptr || hostShape == nullptr)
      THROW_EXCEPTION("LegacyTensorArg requires an NDArray and effective host shape");
    if (ArrayOptions::dataType(hostShape) != const_cast<NDArray*>(array)->dataType())
      THROW_EXCEPTION("LegacyTensorArg effective shape must preserve storage dtype");
    LegacyTensorArg result(array, hostShape, deviceShape, relativeElementOffset);
    result.absoluteElementOffset();  // Check addition before any backend consumes the delta.
    return result;
  }

  LongType absoluteElementOffset() const {
    const auto originalOffset = const_cast<NDArray*>(array)->offset();
    if ((relativeElementOffset > 0 && originalOffset > std::numeric_limits<LongType>::max() - relativeElementOffset) ||
        (relativeElementOffset < 0 && originalOffset < std::numeric_limits<LongType>::min() - relativeElementOffset))
      THROW_EXCEPTION("LegacyTensorArg absolute element offset overflow");
    return originalOffset + relativeElementOffset;
  }

#if !defined(SD_VULKAN)
  // CPU/CUDA accessors already contain array->offset(). Apply ONLY the explicit delta.
  void* hostData() const { return shifted(const_cast<NDArray*>(array)->buffer()); }
  void* deviceData() const { return shifted(const_cast<NDArray*>(array)->specialBuffer()); }
#endif

 private:
  LegacyTensorArg(const NDArray* source, const LongType* hostShape,
                  const LongType* deviceShape, LongType relativeOffset)
      : array(source), hostShapeInfo(hostShape), deviceShapeInfo(deviceShape),
        relativeElementOffset(relativeOffset) {}
#if !defined(SD_VULKAN)
  void* shifted(void* data) const {
    if (data == nullptr || relativeElementOffset == 0) return data;
    const auto width = static_cast<LongType>(const_cast<NDArray*>(array)->sizeOfT());
    const auto maxBytes = static_cast<LongType>(std::numeric_limits<std::ptrdiff_t>::max());
    const auto minBytes = static_cast<LongType>(std::numeric_limits<std::ptrdiff_t>::min());
    if (relativeElementOffset > maxBytes / width || relativeElementOffset < minBytes / width)
      THROW_EXCEPTION("LegacyTensorArg relative byte offset overflows ptrdiff_t");
    return static_cast<char*>(data) + static_cast<std::ptrdiff_t>(relativeElementOffset * width);
  }
#endif
};
}  // namespace sd

// Keep the raw backend ABI unchanged. Vulkan defines metadata overloads in its TUs;
// other artifacts forward through their established shifted-pointer implementations.
#if defined(SD_VULKAN)
#define SD_LEGACY_ADAPTER(NAME, PARAMS, ARGS) static void NAME PARAMS;
#else
#define SD_LEGACY_ADAPTER(NAME, PARAMS, ARGS) static inline void NAME PARAMS { NAME ARGS; }
#endif
#define SD_LEGACY_POINTERS(A) (A).hostData(), const_cast<sd::LongType*>((A).hostShapeInfo), \
                              (A).deviceData(), const_cast<sd::LongType*>((A).deviceShapeInfo)
#endif  // !__JAVACPP_HACK__

/**
 * Native op executioner:
 *
 */

SD_BACKEND_ABI_NAMESPACE_BEGIN

class SD_LIB_EXPORT NativeOpExecutioner {
 public:
  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParams
   * @param result
   * @param resultShapeInfo
   */
  static void execIndexReduceScalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                    const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                    const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParamsVals
   * @param y
   * @param yShapeInfo
   * @param result
   * @param resultShapeInfoBuffer
   * @param dimension
   * @param dimensionLength
   */
  static void execReduce3Scalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                const void *dX, const sd::LongType *dXShapeInfo, void *extraParamsVals, const void *hY,
                                const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                                void *hZ, const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParamsVals
   * @param y
   * @param yShapeInfo
   * @param result
   * @param resultShapeInfo
   */
  static void execReduce3(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                          const void *dX, const sd::LongType *dXShapeInfo, void *extraParamsVals, const void *hY,
                          const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo, void *hZ,
                          const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParamsVals
   * @param y
   * @param yShapeInfo
   * @param result
   * @param resultShapeInfoBuffer
   * @param dimension
   * @param dimensionLength
   */
  static void execReduce3(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                          const void *dX, const sd::LongType *dXShapeInfo, void *extraParamsVals, const void *hY,
                          const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo, void *hZ,
                          const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                          sd::LongType *dimension,
                          sd::LongType dimensionLength, const sd::LongType *xTadOnlyShapeInfo, const sd::LongType *xTadOffsets,
                          const sd::LongType *yTadOnlyShapeInfo, const sd::LongType *yTadOffsets);

  static void execReduce3All(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                             const void *dX, const sd::LongType *dXShapeInfo, void *extraParamsVals, const void *hY,
                             const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo, void *hZ,
                             const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                             sd::LongType *dimension,
                             sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo, const sd::LongType *xOffsets,
                             const sd::LongType *yTadShapeInfo, const sd::LongType *yOffsets);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParams
   * @param result
   * @param resultShapeInfoBuffer
   * @param dimension
   * @param dimensionLength
   */
  static void execIndexReduce(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                              const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                              const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                              sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo,
                              const sd::LongType *tadOffsets);

  /**
   *
   * @param opNum
   * @param x
   * @param xStride
   * @param result
   * @param resultStride
   * @param scalar
   * @param extraParams
   * @param n
   */
  static void execScalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                         const void *dX, const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                         void *dZ, const sd::LongType *dZShapeInfo, const void *hScalar,
                         const sd::LongType *hSscalarShapeInfo, const void *dScalar,
                         const sd::LongType *dSscalarShapeInfo, void *extraParams, bool allowParallelism = true);

  static void execScalarBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                             const void *dX, const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                             void *dZ, const sd::LongType *dZShapeInfo, const void *hScalar,
                             const sd::LongType *hSscalarShapeInfo, const void *dScalar,
                             const sd::LongType *dSscalarShapeInfo, void *extraParams, bool allowParallelism = true);

  static void execScalarInt(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                            const void *dX, const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                            void *dZ, const sd::LongType *dZShapeInfo, const void *hScalar,
                            const sd::LongType *hSscalarShapeInfo, const void *dScalar,
                            const sd::LongType *dSscalarShapeInfo, void *extraParams, bool allowParallelism = true);

  static void execScalar(sd::LaunchContext *lc, int opNum, void const *hX, sd::LongType const *hXShapeInfo,
                         void const *dX, sd::LongType const *dXShapeInfo, void *extraParams, void *hZ,
                         sd::LongType const *hZShapeInfo, void *dZ, sd::LongType const *dZShapeInfo,
                         void const *hScalars, sd::LongType const *hScalarShapeInfo, void const *dScalars,
                         sd::LongType const *dScalarShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength,
                         sd::LongType const *tadShapeInfo, sd::LongType const *tadOffsets,
                         sd::LongType const *tadShapeInfoZ, sd::LongType const *tadOffsetsZ);

  static void execScalarBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                             const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                             const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                             const void *hScalars, const sd::LongType *hScalarShapeInfo, const void *dScalars,
                             const sd::LongType *dScalarShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength,
                             const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets,
                             const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ);

  static void execScalarInt(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                            const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                            const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                            const void *hScalars, const sd::LongType *hScalarShapeInfo, const void *dScalars,
                            const sd::LongType *dScalarShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength,
                            const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets,
                            const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param y
   * @param yShapeInfo
   * @param result
   * @param resultShapeInfo
   * @param dimension
   * @param dimensionLength
   */
  static void execBroadcast(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                            const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                            const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo, void *hZ,
                            const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                            sd::LongType *dimension,
                            sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets,
                            const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ);

  static void execBroadcast(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                            const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                            const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo, void *hZ,
                            const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  static void execInverseBroadcast(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                   const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                                   const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                                   void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
                                   const sd::LongType *dZShapeInfo,
                                   sd::LongType *dimension, sd::LongType dimensionLength,
                                   const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets,
                                   const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ);

  static void execBroadcastBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                                const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                                void *hZ, const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                void *extraParams, sd::LongType *dimension, sd::LongType dimensionLength,
                                const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets,
                                const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ);

  static void execBroadcastBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                                const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                                void *hZ, const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                void *extraParams);

  static void execInverseBroadcastBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                       const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                                       const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                                       void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
                                       const sd::LongType *dZShapeInfo, void *extraParams,
                                       sd::LongType *dimension,
                                       sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo,
                                       const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ,
                                       const sd::LongType *tadOffsetsZ);

  static void execBroadcastInt(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                               const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                               const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                               void *hZ, const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                               sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo,
                               const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ,
                               const sd::LongType *tadOffsetsZ);

  static void execBroadcastInt(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                               const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                               const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                               void *hZ, const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  static void execInverseBroadcastInt(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                      const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                                      const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                                      void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
                                      const sd::LongType *dZShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength,
                                      const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets,
                                      const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ);

  /**
   *
   * @param opNum
   * @param dx
   * @param xStride
   * @param y
   * @param yStride
   * @param result
   * @param resultStride
   * @param extraParams
   * @param n
   */
  static void execPairwiseTransform(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                    const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                                    const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                                    void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
                                    const sd::LongType *dZShapeInfo, void *extraParams);

  static void execPairwiseBoolTransform(sd::LaunchContext *lc, int opNum, const void *hX,
                                        const sd::LongType *hXShapeInfo, const void *dX,
                                        const sd::LongType *dXShapeInfo, const void *hY,
                                        const sd::LongType *hYShapeInfo, const void *dY,
                                        const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                        void *dZ, const sd::LongType *dZShapeInfo, void *extraParams);

  static void execPairwiseIntTransform(sd::LaunchContext *lc, int opNum, const void *hX,
                                       const sd::LongType *hXShapeInfo, const void *dX, const sd::LongType *dXShapeInfo,
                                       const void *hY, const sd::LongType *hYShapeInfo, const void *dY,
                                       const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                       void *dZ, const sd::LongType *dZShapeInfo, void *extraParams);

  /**
   *
   * @param opNum
   * @param dx
   * @param xStride
   * @param result
   * @param resultStride
   * @param extraParams
   * @param n
   */
  static void execTransformFloat(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                 const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                                 const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                 void *extraParams);

  static void execTransformAny(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                               const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                               const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                               void *extraParams, bool allowParallelism);

  static void execTransformStrict(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                  const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                                  const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                  void *extraParams);

  static void execTransformSame(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                                const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                void *extraParams, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets);

  static void execTransformBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                                const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                void *extraParams);
  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParams
   * @param result
   * @param resultShapeInfo
   */
  static void execReduceFloat(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                              const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                              const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                              sd::LongType *dimension, sd::LongType dimensionLength);

  static void execReduceSame(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                             const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                             const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                             sd::LongType *dimension,
                             sd::LongType dimensionLength);

  static void execReduceBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                             const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                             const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                             sd::LongType *dimension,
                             sd::LongType dimensionLength);

  static void execReduceLong(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                             const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                             const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                             sd::LongType *dimension,
                             sd::LongType dimensionLength);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParams
   * @return
   */
  static void execReduceFloatScalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                    const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                    const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  static void execReduceBoolScalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                   const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                   const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  static void execReduceSameScalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                   const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                   const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  static void execReduceLongScalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                   const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                   const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  static void execReduce3TAD(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                             const void *dX, const sd::LongType *dXShapeInfo, void *extraParamsVals, const void *hY,
                             const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo, void *hZ,
                             const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                             sd::LongType *dimension,
                             sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets,
                             const sd::LongType *yTadShapeInfo, const sd::LongType *yTadOffsets);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParams
   * @param result
   * @param resultShapeInfoBuffer
   * @param dimension
   * @param dimensionLength
   */
  static void execSummaryStats(sd::LaunchContext *lc, int opNum, const void *hX, sd::LongType *hXShapeInfo,
                               const void *dX, sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                               sd::LongType *hZShapeInfo, void *dZ, sd::LongType *dZShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, sd::LongType *tadShapeInfo,
                               sd::LongType *tadOffsets, bool biasCorrected);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParams
   * @param result
   * @param resultShapeInfo
   */
  static void execSummaryStats(sd::LaunchContext *lc, int opNum, const void *hX, sd::LongType *hXShapeInfo,
                               const void *dX, sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                               sd::LongType *hZShapeInfo, void *dZ, sd::LongType *dZShapeInfo,
                               bool biasCorrected);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParams
   * @param result
   * @param resultShapeInfo
   */
  static void execSummaryStatsScalar(sd::LaunchContext *lc, int opNum, const void *hX, sd::LongType *hXShapeInfo,
                                     const void *dX, sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                     sd::LongType *hZShapeInfo, void *dZ, sd::LongType *dZShapeInfo,
                                     bool biasCorrected);

  static void execRandom(sd::LaunchContext *lc, int opNum, sd::Pointer state, void *hZ,
                         const sd::LongType *hZShapeBuffer, void *dZ, const sd::LongType *dZShapeBuffer,
                         void *extraArguments);

  static void execRandom(sd::LaunchContext *lc, int opNum, sd::Pointer state, const void *hX,
                         const sd::LongType *hXShapeBuffer, const void *dX, const sd::LongType *dXShapeBuffer, void *hZ,
                         const sd::LongType *hZShapeBuffer, void *dZ, const sd::LongType *dZShapeBuffer,
                         void *extraArguments);

  static void execRandom(sd::LaunchContext *lc, int opNum, sd::Pointer state, const void *hX,
                         const sd::LongType *hXShapeBuffer, const void *dX, const sd::LongType *dXShapeBuffer,
                         const void *hY, const sd::LongType *hYShapeBuffer, const void *dY,
                         const sd::LongType *dYShapeBuffer, void *hZ, const sd::LongType *hZShapeBuffer, void *dZ,
                         const sd::LongType *dZShapeBuffer, void *extraArguments);

#ifndef __JAVACPP_HACK__
  // Additive metadata-bearing overloads: one carrier replaces each four-pointer tensor tuple.
#define SD_LEGACY_UNARY(NAME) \
  SD_LEGACY_ADAPTER(NAME, (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, \
                          const sd::LegacyTensorArg& z, void* extraParams), \
                         (lc, opNum, SD_LEGACY_POINTERS(x), SD_LEGACY_POINTERS(z), extraParams))
  SD_LEGACY_UNARY(execTransformFloat)
  SD_LEGACY_UNARY(execTransformStrict)
  SD_LEGACY_UNARY(execTransformBool)
#undef SD_LEGACY_UNARY
  SD_LEGACY_ADAPTER(execTransformAny,
      (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, const sd::LegacyTensorArg& z,
       void* extraParams, bool allowParallelism),
      (lc, opNum, SD_LEGACY_POINTERS(x), SD_LEGACY_POINTERS(z), extraParams, allowParallelism))
  SD_LEGACY_ADAPTER(execTransformSame,
      (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, const sd::LegacyTensorArg& z,
       void* extraParams, const sd::LongType* tadShapeInfo, const sd::LongType* tadOffsets),
      (lc, opNum, SD_LEGACY_POINTERS(x), SD_LEGACY_POINTERS(z), extraParams, tadShapeInfo, tadOffsets))
#define SD_LEGACY_PAIRWISE(NAME) \
  SD_LEGACY_ADAPTER(NAME, (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, \
                          const sd::LegacyTensorArg& y, const sd::LegacyTensorArg& z, void* extraParams), \
                         (lc, opNum, SD_LEGACY_POINTERS(x), SD_LEGACY_POINTERS(y), \
                          SD_LEGACY_POINTERS(z), extraParams))
  SD_LEGACY_PAIRWISE(execPairwiseTransform)
  SD_LEGACY_PAIRWISE(execPairwiseBoolTransform)
  SD_LEGACY_PAIRWISE(execPairwiseIntTransform)
#undef SD_LEGACY_PAIRWISE
#define SD_LEGACY_SCALAR(NAME) \
  SD_LEGACY_ADAPTER(NAME, (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, \
                          const sd::LegacyTensorArg& z, const sd::LegacyTensorArg& scalar, \
                          void* extraParams, bool allowParallelism = true), \
                         (lc, opNum, SD_LEGACY_POINTERS(x), SD_LEGACY_POINTERS(z), \
                          SD_LEGACY_POINTERS(scalar), extraParams, allowParallelism)) \
  SD_LEGACY_ADAPTER(NAME, (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, \
                          void* extraParams, const sd::LegacyTensorArg& z, const sd::LegacyTensorArg& scalars, \
                          sd::LongType* dimension, sd::LongType dimensionLength, \
                          const sd::LongType* tadShapeInfo, const sd::LongType* tadOffsets, \
                          const sd::LongType* tadShapeInfoZ, const sd::LongType* tadOffsetsZ), \
                         (lc, opNum, SD_LEGACY_POINTERS(x), extraParams, SD_LEGACY_POINTERS(z), \
                          SD_LEGACY_POINTERS(scalars), dimension, dimensionLength, \
                          tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ))
  SD_LEGACY_SCALAR(execScalar)
  SD_LEGACY_SCALAR(execScalarBool)
  SD_LEGACY_SCALAR(execScalarInt)
#undef SD_LEGACY_SCALAR
#define SD_LEGACY_BROADCAST(NAME) \
  SD_LEGACY_ADAPTER(NAME, (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, \
                          const sd::LegacyTensorArg& y, const sd::LegacyTensorArg& z, \
                          sd::LongType* dimension, sd::LongType dimensionLength, \
                          const sd::LongType* tadOnlyShapeInfo, const sd::LongType* tadOffsets, \
                          const sd::LongType* tadOnlyShapeInfoZ, const sd::LongType* tadOffsetsZ), \
                         (lc, opNum, SD_LEGACY_POINTERS(x), SD_LEGACY_POINTERS(y), SD_LEGACY_POINTERS(z), \
                          dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ))
  SD_LEGACY_BROADCAST(execBroadcast)
  SD_LEGACY_BROADCAST(execInverseBroadcast)
  SD_LEGACY_BROADCAST(execBroadcastInt)
  SD_LEGACY_BROADCAST(execInverseBroadcastInt)
#undef SD_LEGACY_BROADCAST
#define SD_LEGACY_BROADCAST_SIMPLE(NAME) \
  SD_LEGACY_ADAPTER(NAME, (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, \
                          const sd::LegacyTensorArg& y, const sd::LegacyTensorArg& z), \
                         (lc, opNum, SD_LEGACY_POINTERS(x), SD_LEGACY_POINTERS(y), SD_LEGACY_POINTERS(z)))
  SD_LEGACY_BROADCAST_SIMPLE(execBroadcast)
  SD_LEGACY_BROADCAST_SIMPLE(execBroadcastInt)
#undef SD_LEGACY_BROADCAST_SIMPLE
#define SD_LEGACY_BROADCAST_BOOL(NAME) \
  SD_LEGACY_ADAPTER(NAME, (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, \
                          const sd::LegacyTensorArg& y, const sd::LegacyTensorArg& z, void* extraParams, \
                          sd::LongType* dimension, sd::LongType dimensionLength, \
                          const sd::LongType* tadOnlyShapeInfo, const sd::LongType* tadOffsets, \
                          const sd::LongType* tadOnlyShapeInfoZ, const sd::LongType* tadOffsetsZ), \
                         (lc, opNum, SD_LEGACY_POINTERS(x), SD_LEGACY_POINTERS(y), SD_LEGACY_POINTERS(z), \
                          extraParams, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, \
                          tadOnlyShapeInfoZ, tadOffsetsZ))
  SD_LEGACY_BROADCAST_BOOL(execBroadcastBool)
  SD_LEGACY_BROADCAST_BOOL(execInverseBroadcastBool)
#undef SD_LEGACY_BROADCAST_BOOL
  SD_LEGACY_ADAPTER(execBroadcastBool,
      (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, const sd::LegacyTensorArg& y,
       const sd::LegacyTensorArg& z, void* extraParams),
      (lc, opNum, SD_LEGACY_POINTERS(x), SD_LEGACY_POINTERS(y), SD_LEGACY_POINTERS(z), extraParams))
#define SD_LEGACY_REDUCE(NAME) \
  SD_LEGACY_ADAPTER(NAME, (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, \
                          void* extraParams, const sd::LegacyTensorArg& z, \
                          sd::LongType* dimension, sd::LongType dimensionLength), \
                         (lc, opNum, SD_LEGACY_POINTERS(x), extraParams, SD_LEGACY_POINTERS(z), \
                          dimension, dimensionLength))
  SD_LEGACY_REDUCE(execReduceFloat)
  SD_LEGACY_REDUCE(execReduceSame)
  SD_LEGACY_REDUCE(execReduceBool)
  SD_LEGACY_REDUCE(execReduceLong)
#undef SD_LEGACY_REDUCE
#define SD_LEGACY_REDUCE_SCALAR(NAME) \
  SD_LEGACY_ADAPTER(NAME, (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, \
                          void* extraParams, const sd::LegacyTensorArg& z), \
                         (lc, opNum, SD_LEGACY_POINTERS(x), extraParams, SD_LEGACY_POINTERS(z)))
  SD_LEGACY_REDUCE_SCALAR(execReduceFloatScalar)
  SD_LEGACY_REDUCE_SCALAR(execReduceSameScalar)
  SD_LEGACY_REDUCE_SCALAR(execReduceBoolScalar)
  SD_LEGACY_REDUCE_SCALAR(execReduceLongScalar)
  SD_LEGACY_REDUCE_SCALAR(execIndexReduceScalar)
#undef SD_LEGACY_REDUCE_SCALAR
  SD_LEGACY_ADAPTER(execIndexReduce,
      (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, void* extraParams,
       const sd::LegacyTensorArg& z, sd::LongType* dimension, sd::LongType dimensionLength,
       const sd::LongType* tadShapeInfo, const sd::LongType* tadOffsets),
      (lc, opNum, SD_LEGACY_POINTERS(x), extraParams, SD_LEGACY_POINTERS(z), dimension,
       dimensionLength, tadShapeInfo, tadOffsets))
#define SD_LEGACY_REDUCE3_SIMPLE(NAME) \
  SD_LEGACY_ADAPTER(NAME, (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, \
                          void* extraParamsVals, const sd::LegacyTensorArg& y, const sd::LegacyTensorArg& z), \
                         (lc, opNum, SD_LEGACY_POINTERS(x), extraParamsVals, SD_LEGACY_POINTERS(y), \
                          SD_LEGACY_POINTERS(z)))
  SD_LEGACY_REDUCE3_SIMPLE(execReduce3)
  SD_LEGACY_REDUCE3_SIMPLE(execReduce3Scalar)
#undef SD_LEGACY_REDUCE3_SIMPLE
#define SD_LEGACY_REDUCE3(NAME) \
  SD_LEGACY_ADAPTER(NAME, (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, \
                          void* extraParamsVals, const sd::LegacyTensorArg& y, const sd::LegacyTensorArg& z, \
                          sd::LongType* dimension, sd::LongType dimensionLength, \
                          const sd::LongType* xTadShapeInfo, const sd::LongType* xTadOffsets, \
                          const sd::LongType* yTadShapeInfo, const sd::LongType* yTadOffsets), \
                         (lc, opNum, SD_LEGACY_POINTERS(x), extraParamsVals, SD_LEGACY_POINTERS(y), \
                          SD_LEGACY_POINTERS(z), dimension, dimensionLength, xTadShapeInfo, xTadOffsets, \
                          yTadShapeInfo, yTadOffsets))
  SD_LEGACY_REDUCE3(execReduce3)
  SD_LEGACY_REDUCE3(execReduce3All)
  SD_LEGACY_REDUCE3(execReduce3TAD)
#undef SD_LEGACY_REDUCE3
  SD_LEGACY_ADAPTER(execSummaryStats,
      (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, void* extraParams,
       const sd::LegacyTensorArg& z, sd::LongType* dimension, sd::LongType dimensionLength,
       sd::LongType* tadShapeInfo, sd::LongType* tadOffsets, bool biasCorrected),
      (lc, opNum, SD_LEGACY_POINTERS(x), extraParams, SD_LEGACY_POINTERS(z), dimension,
       dimensionLength, tadShapeInfo, tadOffsets, biasCorrected))
#define SD_LEGACY_SUMMARY(NAME) \
  SD_LEGACY_ADAPTER(NAME, (sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, \
                          void* extraParams, const sd::LegacyTensorArg& z, bool biasCorrected), \
                         (lc, opNum, SD_LEGACY_POINTERS(x), extraParams, SD_LEGACY_POINTERS(z), biasCorrected))
  SD_LEGACY_SUMMARY(execSummaryStats)
  SD_LEGACY_SUMMARY(execSummaryStatsScalar)
#undef SD_LEGACY_SUMMARY
  SD_LEGACY_ADAPTER(execRandom,
      (sd::LaunchContext* lc, int opNum, sd::Pointer state, const sd::LegacyTensorArg& z, void* extraArguments),
      (lc, opNum, state, SD_LEGACY_POINTERS(z), extraArguments))
  SD_LEGACY_ADAPTER(execRandom,
      (sd::LaunchContext* lc, int opNum, sd::Pointer state, const sd::LegacyTensorArg& x,
       const sd::LegacyTensorArg& z, void* extraArguments),
      (lc, opNum, state, SD_LEGACY_POINTERS(x), SD_LEGACY_POINTERS(z), extraArguments))
  SD_LEGACY_ADAPTER(execRandom,
      (sd::LaunchContext* lc, int opNum, sd::Pointer state, const sd::LegacyTensorArg& x,
       const sd::LegacyTensorArg& y, const sd::LegacyTensorArg& z, void* extraArguments),
      (lc, opNum, state, SD_LEGACY_POINTERS(x), SD_LEGACY_POINTERS(y), SD_LEGACY_POINTERS(z), extraArguments))
#undef SD_LEGACY_ADAPTER
#undef SD_LEGACY_POINTERS
#endif  // !__JAVACPP_HACK__

  // Inline implementations to avoid source location exhaustion in separate compilation units
  static inline void execSort(sd::NDArray *x, bool descending) {
    auto xType = x->dataType();
    BUILD_SINGLE_SELECTOR(xType, sd::SpecialMethods, ::sortGeneric(x, descending), SD_COMMON_TYPES);
  }

  static inline void execSort(sd::NDArray *x, sd::LongType *dimension, sd::LongType dimensionLength,
                               bool descending) {
    auto xType = x->dataType();
    BUILD_SINGLE_SELECTOR(xType, sd::SpecialMethods,
                          ::sortTadGeneric(x, dimension, dimensionLength, descending),
                          SD_COMMON_TYPES);
  }




};

SD_BACKEND_ABI_NAMESPACE_END
SD_BACKEND_ABI_ALIAS(NativeOpExecutioner)

// Preserve the historical unqualified NativeOpExecutioner spelling while the
// actual class and its backend-varying method symbols use the backend ABI namespace.
using NativeOpExecutioner = sd::NativeOpExecutioner;

#endif  // NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H
