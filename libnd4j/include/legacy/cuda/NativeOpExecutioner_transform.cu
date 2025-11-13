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
#include <array/ConstantDataBuffer.h>
#include <array/DataTypeUtils.h>
#include <array/ShapeDescriptor.h>
#include <exceptions/cuda_exception.h>
#include <exceptions/datatype_exception.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/DebugHelper.h>
#include <helpers/PointersManager.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/broadcasting.h>
#include <loops/broadcasting_bool.h>
#include <loops/broadcasting_int.h>
#include <loops/indexreduce.h>
#include <loops/pairwise_bool.h>
#include <loops/pairwise_int.h>
#include <loops/pairwise_transform.h>
#include <loops/random.h>
#include <loops/reduce3.h>
#include <loops/reduce_bool.h>
#include <loops/reduce_float.h>
#include <loops/reduce_long.h>
#include <loops/reduce_same.h>
#include <loops/scalar.h>
#include <loops/scalar_bool.h>
#include <loops/scalar_int.h>
#include <loops/special_kernels.h>
#include <loops/summarystatsreduce.h>
#include <loops/transform_any.h>
#include <loops/transform_bool.h>
#include <loops/transform_float.h>
#include <loops/transform_same.h>
#include <loops/transform_strict.h>
#include <system/op_boilerplate.h>
#include <helpers/ConstantTadHelper.h>
#include <system/selective_rendering.h>

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformSame(sd::LaunchContext* lc, int opNum, void const* hX,
                                            sd::LongType const* hXShapeInfo, void const* dX, sd::LongType const* dXShapeInfo,
                                            void* hZ, sd::LongType const* hZShapeInfo, void* dZ,
                                            sd::LongType const* dZShapeInfo, void* extraParams,
                                            sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets) {
  auto stream = lc->getCudaStream();

  auto xRank = shape::rank(hXShapeInfo);
  auto zRank = shape::rank(hZShapeInfo);
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execTransformSame:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }

  if (xType != zType) {
    THROW_EXCEPTION("NativeOpExecutioner::execTransformSame requires X & Z to have same type");
  }

  dim3 launchDims = getLaunchDims("transformScan");
  BUILD_SINGLE_SELECTOR(xType, functions::transform::TransformSame,
                        ::executeTransformShaped(launchDims, stream, opNum, dX, dXShapeInfo, xRank, extraParams, dZ,
                                                 dZShapeInfo, zRank, nullptr, nullptr, tadShapeInfo, tadOffsets),
                        SD_COMMON_TYPES);
}


void NativeOpExecutioner::execTransformFloat(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                             const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                                             const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                             void *extraParams) {
  auto stream = lc->getCudaStream();

  auto xRank = shape::rank(hXShapeInfo);
  auto zRank = shape::rank(hZShapeInfo);
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execTransformFloat:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }

  if (xType != zType) {
    THROW_EXCEPTION("NativeOpExecutioner::execTransformSame requires X & Z to have same type");
  }

  dim3 launchDims = getLaunchDims("transformScan");
  BUILD_DOUBLE_SELECTOR(xType,xType, functions::transform::TransformFloat,
                        ::executeTransformShaped(launchDims, stream,  opNum, dX,
                                                 dXShapeInfo, xRank,extraParams, dZ,
                                                 dZShapeInfo, zRank,
                                                 nullptr,nullptr,
                                                 nullptr,nullptr),
                        SD_COMMON_TYPES,SD_COMMON_TYPES);
}

void NativeOpExecutioner::execTransformStrict(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                              const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                                              const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                              void *extraParams) {
  auto stream = lc->getCudaStream();
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execTransformStrict:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }

  if (xType != zType) {
    THROW_EXCEPTION("NativeOpExecutioner::execTransformStrict requires X & Z to have same type");
  }

  dim3 launchDims = getLaunchDims("transformScan");
  BUILD_SINGLE_SELECTOR(xType, functions::transform::TransformStrict,
                        ::executeTransformShaped(launchDims, stream, opNum, dX, dXShapeInfo, shape::rank(hXShapeInfo), extraParams, dZ, dZShapeInfo, shape::rank(hZShapeInfo), nullptr, nullptr, nullptr, nullptr),
                        SD_COMMON_TYPES);

}


////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformAny(sd::LaunchContext *lc, int opNum, const void *hX,
                                           const sd::LongType *hXShapeInfo, const void *dX,
                                           const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                           void *dZ, const sd::LongType *dZShapeInfo, void *extraParams,
                                           bool allowParallelism) {
  auto stream = lc->getCudaStream();

  auto xRank = shape::rank(hXShapeInfo);
  auto zRank = shape::rank(hZShapeInfo);
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execTransformAny:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  dim3 launchDims = getLaunchDims("transformScan");
  if (sd::DataTypeUtils::isS(xType)) {
#if defined(HAS_UTF8) || defined(HAS_UTF16) || defined(HAS_UTF32)
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformAny,
                          ::executeTransformShaped(launchDims, stream, opNum, dX, dXShapeInfo, xRank, extraParams, dZ,
                                                   dZShapeInfo, zRank, nullptr, nullptr, nullptr, nullptr),
                          SD_STRING_TYPES, SD_STRING_TYPES);
#endif
  } else {
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformAny,
                          ::executeTransformShaped(launchDims, stream, opNum, dX, dXShapeInfo, xRank, extraParams, dZ,
                                                   dZShapeInfo, zRank, nullptr, nullptr, nullptr, nullptr),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);
  }
}



////////////////////////////////////////////////////////////////////////
