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
void NativeOpExecutioner::execBroadcastBool(sd::LaunchContext* lc, int opNum, void const* hX, sd::LongType const* hXShapeInfo,
                                            void const* dX, sd::LongType const* dXShapeInfo, void const* hY,
                                            sd::LongType const* hYShapeInfo, void const* dY, sd::LongType const* dYShapeInfo,
                                            void* hZ, sd::LongType const* hZShapeInfo, void* dZ,
                                            sd::LongType const* dZShapeInfo, void* extraParams, sd::LongType* dimension,
                                            sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo,
                                            sd::LongType const* tadOffsets, sd::LongType const* tadOnlyShapeInfoZ,
                                            sd::LongType const* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execBroadcastBool:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }

  if (!sd::DataTypeUtils::isB(zType))
    THROW_EXCEPTION("NativeOpExecutioner::execBroadcastBool requires Z operand to have BOOL type");

  if (yType != xType)
    THROW_EXCEPTION("NativeOpExecutioner::execBroadcastBool requires both X & Y operands to have same type");

  if (sd::Environment::getInstance().isDebugAndVerbose()) printf("F3B opType:[%i]\n", opNum);

  dim3 launchDims = getLaunchDims("broadcast");
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::broadcast::BroadcastBool,
      ::execBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraParams,
                      dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
      SD_COMMON_TYPES, SD_BOOL_TYPES)
#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcastBool(sd::LaunchContext* lc, const int opNum, const void* hX,
                                            const sd::LongType* hXShapeInfo, const void* dX, const sd::LongType* dXShapeInfo,
                                            const void* hY, const sd::LongType* hYShapeInfo, const void* dY,
                                            const sd::LongType* dYShapeInfo, void* hZ, const sd::LongType* hZShapeInfo,
                                            void* dZ, const sd::LongType* dZShapeInfo, void* extraParams) {
  auto stream = lc->getCudaStream();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execBroadcastBool:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }
  dim3 launchDims = getLaunchDims("broadcastBool");
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::broadcast::BroadcastBool,
      ::execBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraParams),
      SD_COMMON_TYPES, SD_BOOL_TYPES);
}

void NativeOpExecutioner::execInverseBroadcastBool(
    sd::LaunchContext* lc, int opNum, void const* hX, sd::LongType const* hXShapeInfo, void const* dX,
    sd::LongType const* dXShapeInfo, void const* hY, sd::LongType const* hYShapeInfo, void const* dY,
    sd::LongType const* dYShapeInfo, void* hZ, sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo,
    void* extraParams, sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo,
    sd::LongType const* tadOffsets, sd::LongType const* tadOnlyShapeInfoZ, sd::LongType const* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execInverseBroadcastBool:: unable to execute on strings. Please write logic higher level "
        "in each op for the string data type.")
  }
  if (!sd::DataTypeUtils::isB(zType))
    THROW_EXCEPTION("NativeOpExecutioner::execBroadcastBool requires Z operand to have BOOL type");

  if (yType != xType)
    THROW_EXCEPTION("NativeOpExecutioner::execBroadcastBool requires both X & Y operands to have same type");

  dim3 launchDims = getLaunchDims("broadcastBool");
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::broadcast::BroadcastBool,
      ::execInverseBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraParams,
                             dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
      SD_COMMON_TYPES, SD_BOOL_TYPES)
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcastInt(sd::LaunchContext* lc, int opNum, void const* hX,
                                           sd::LongType const* hXShapeInfo, void const* dX, sd::LongType const* dXShapeInfo,
                                           void const* hY, sd::LongType const* hYShapeInfo, void const* dY,
                                           sd::LongType const* dYShapeInfo, void* hZ, sd::LongType const* hZShapeInfo, void* dZ,
                                           sd::LongType const* dZShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength,
                                           sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets,
                                           sd::LongType const* tadOnlyShapeInfoZ, sd::LongType const* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execBroadcastInt:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  if (!sd::DataTypeUtils::isZ(zType))
    THROW_EXCEPTION("NativeOpExecutioner::execBroadcastInt requires Z operand to have INT type");

  if (yType != xType || zType != xType)
    THROW_EXCEPTION("NativeOpExecutioner::execBroadcastInt requires both X & Y operands to have same type");

  dim3 launchDims = getLaunchDims("broadcastInt");

  BUILD_SINGLE_SELECTOR(
      xType, functions::broadcast::BroadcastInt,
      ::execBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, dimension,
                      dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
      SD_INTEGER_TYPES)
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcastInt(sd::LaunchContext* lc, const int opNum, const void* hX,
                                           const sd::LongType* hXShapeInfo, const void* dX, const sd::LongType* dXShapeInfo,
                                           const void* hY, const sd::LongType* hYShapeInfo, const void* dY,
                                           const sd::LongType* dYShapeInfo, void* hZ, const sd::LongType* hZShapeInfo, void* dZ,
                                           const sd::LongType* dZShapeInfo) {
  auto stream = lc->getCudaStream();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOPExecutioner::execBroadcastInt:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }

  if (!sd::DataTypeUtils::isZ(zType))
    THROW_EXCEPTION("NativeOpExecutioner::execBroadcastInt requires Z operand to have INT type");

  if (yType != xType || zType != xType)
    THROW_EXCEPTION("NativeOpExecutioner::execBroadcastInt requires both X & Y operands to have same type");

  dim3 launchDims = getLaunchDims("broadcastInt");
  // shared memory

  BUILD_SINGLE_SELECTOR(xType, functions::broadcast::BroadcastInt,
                        ::execBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo),
                        SD_INTEGER_TYPES)
}

void NativeOpExecutioner::execInverseBroadcastInt(
    sd::LaunchContext* lc, int opNum, void const* hX, sd::LongType const* hXShapeInfo, void const* dX,
    sd::LongType const* dXShapeInfo, void const* hY, sd::LongType const* hYShapeInfo, void const* dY,
    sd::LongType const* dYShapeInfo, void* hZ, sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo,
    sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo, sd::LongType const* tadOffsets,
    sd::LongType const* tadOnlyShapeInfoZ, sd::LongType const* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execInverseBroadcastInt:: unable to execute on strings. Please write logic higher level "
        "in each op for the string data type.")
  }
  if (!sd::DataTypeUtils::isZ(zType))
    THROW_EXCEPTION("NativeOpExecutioner::execInverseBroadcastInt requires Z operand to have INT type");

  if (yType != xType || zType != xType)
    THROW_EXCEPTION("NativeOpExecutioner::execInverseBroadcastInt requires both X & Y operands to have same type");

  if (sd::Environment::getInstance().isDebugAndVerbose()) printf("F3BI opType:[%i]\n", opNum);

  dim3 launchDims = getLaunchDims("broadcastInt");

  BUILD_SINGLE_SELECTOR(
      xType, functions::broadcast::BroadcastInt,
      ::execInverseBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, dimension,
                             dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
      SD_INTEGER_TYPES)
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param dY
 * @param dYShapeInfo
 * @param dZ
 * @param dZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void NativeOpExecutioner::execBroadcast(sd::LaunchContext* lc, int opNum, void const* hX, sd::LongType const* hXShapeInfo,
                                        void const* dX, sd::LongType const* dXShapeInfo, void const* hY,
                                        sd::LongType const* hYShapeInfo, void const* dY, sd::LongType const* dYShapeInfo,
                                        void* hZ, sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo,
                                        sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo,
                                        sd::LongType const* tadOffsets, sd::LongType const* tadOnlyShapeInfoZ,
                                        sd::LongType const* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOPExecutioner::execBroadcast:: unable to execute on strings. Please write logic higher level in each op "
        "for the string data type.")
  }
  dim3 launchDims = getLaunchDims("broadcast");

  BUILD_SINGLE_SELECTOR_THRICE(
      xType, functions::broadcast::Broadcast,
      ::execBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, dimension,
                      dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
      SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcast(sd::LaunchContext* lc, const int opNum, const void* hX,
                                        const sd::LongType* hXShapeInfo, const void* dX, const sd::LongType* dXShapeInfo,
                                        const void* hY, const sd::LongType* hYShapeInfo, const void* dY,
                                        const sd::LongType* dYShapeInfo, void* hZ, const sd::LongType* hZShapeInfo, void* dZ,
                                        const sd::LongType* dZShapeInfo) {
  auto stream = lc->getCudaStream();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execBroadcast:: unable to execute on strings. Please write logic higher level in each op "
        "for the string data type.")
  }

  dim3 launchDims = getLaunchDims("broadcast");
  // shared memory

  BUILD_SINGLE_SELECTOR_THRICE(
      xType, functions::broadcast::Broadcast,
      ::execBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo), SD_COMMON_TYPES);
}

void NativeOpExecutioner::execInverseBroadcast(sd::LaunchContext* lc, int opNum, void const* hX,
                                               sd::LongType const* hXShapeInfo, void const* dX, sd::LongType const* dXShapeInfo,
                                               void const* hY, sd::LongType const* hYShapeInfo, void const* dY,
                                               sd::LongType const* dYShapeInfo, void* hZ, sd::LongType const* hZShapeInfo,
                                               void* dZ, sd::LongType const* dZShapeInfo, sd::LongType* dimension,
                                               sd::LongType dimensionLength, sd::LongType const* tadOnlyShapeInfo,
                                               sd::LongType const* tadOffsets, sd::LongType const* tadOnlyShapeInfoZ,
                                               sd::LongType const* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execInverseBroadcast:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }

  dim3 launchDims = getLaunchDims("broadcast");

  BUILD_SINGLE_SELECTOR_THRICE(
      xType, functions::broadcast::Broadcast,
      ::execInverseBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, dimension,
                             dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
      SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////
