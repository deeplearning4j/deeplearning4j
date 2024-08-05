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


using namespace sd;

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execPairwiseTransform(LaunchContext* lc, int opNum, void const* hX,
                                                LongType const* hXShapeInfo, void const* dX,
                                                LongType const* dXShapeInfo, void const* hY,
                                                LongType const* hYShapeInfo, void const* dY,
                                                LongType const* dYShapeInfo, void* hZ, LongType const* hZShapeInfo,
                                                void* dZ, LongType const* dZShapeInfo, void* extraParams) {
  auto stream = lc->getCudaStream();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hYShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    std::string errorMessage;
    errorMessage +=
        "NativeOpExecutioner::execPairwiseTransform:: unable to execute on strings. Please write logic "
        "higher level in each op for the string data type.";
    errorMessage += "X type: ";
    errorMessage += DataTypeUtils::asString(xType);
    errorMessage += "Y type: ";
    errorMessage += DataTypeUtils::asString(yType);
    errorMessage += "Z type: ";
    errorMessage += DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (xType != zType && yType != zType) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execPairwiseTransform both operands must have same data type";
    errorMessage += "X type: ";
    errorMessage += DataTypeUtils::asString(xType);
    errorMessage += "Y type: ";
    errorMessage += DataTypeUtils::asString(yType);
    errorMessage += "Z type: ";
    errorMessage += DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (lc == nullptr) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execPairwiseTransform: launch context cannot be nullptr !";
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (stream == nullptr) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execPairwiseTransform: CUDA stream cannot be nullptr !";
    THROW_EXCEPTION(errorMessage.c_str());
  }
  dim3 launchDims = getLaunchDims("pairwiseTransforms");

#ifdef SD_EXPERIMENTAL_ENABLED
  BUILD_PAIRWISE_SELECTOR(
      xType, yType, zType, functions::pairwise_transforms::PairWiseTransform,
      ::executeCudaShaped(launchDims, stream, opType, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraParams),
      SD_COMMON_TYPES, SD_COMMON_TYPES)
#else
  BUILD_SINGLE_SELECTOR_THRICE(
      xType, functions::pairwise_transforms::PairWiseTransform,
      ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraParams),
      SD_COMMON_TYPES)
#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execPairwiseBoolTransform(LaunchContext* lc, int opNum, void const* hX,
                                                    LongType const* hXShapeInfo, void const* dX,
                                                    LongType const* dXShapeInfo, void const* hY,
                                                    LongType const* hYShapeInfo, void const* dY,
                                                    LongType const* dYShapeInfo, void* hZ, LongType const* hZShapeInfo,
                                                    void* dZ, LongType const* dZShapeInfo, void* extraParams) {
  auto stream = lc->getCudaStream();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hYShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execPairwiseBoolTransform:: unable to execute on strings. Please write logic higher "
        "level in each op for the string data type.")
  }

  if (!DataTypeUtils::isB(zType)) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execPairwiseBoolTransform requires Z operand to have BOOL type";
    errorMessage += "X type: ";
    errorMessage += DataTypeUtils::asString(xType);
    errorMessage += "Y type: ";
    errorMessage += DataTypeUtils::asString(yType);
    errorMessage += "Z type: ";
    errorMessage += DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (yType != xType) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execPairwiseBoolTransform both operands must have same data type";
    errorMessage += "X type: ";
    errorMessage += DataTypeUtils::asString(xType);
    errorMessage += "Y type: ";
    errorMessage += DataTypeUtils::asString(yType);

    THROW_EXCEPTION(errorMessage.c_str());
  }
  dim3 launchDims = getLaunchDims("pairwiseTransforms");

  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::pairwise_transforms::PairWiseBoolTransform,
      ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraParams),
      SD_COMMON_TYPES, SD_BOOL_TYPES)
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execPairwiseIntTransform(LaunchContext* lc, int opNum, void const* hX,
                                                   LongType const* hXShapeInfo, void const* dX,
                                                   LongType const* dXShapeInfo, void const* hY,
                                                   LongType const* hYShapeInfo, void const* dY,
                                                   LongType const* dYShapeInfo, void* hZ, LongType const* hZShapeInfo,
                                                   void* dZ, LongType const* dZShapeInfo, void* extraParams) {
  auto stream = lc->getCudaStream();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hYShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    std::string errorMessage;
    errorMessage +=
        "NativeOpExecutioner::execPairwiseIntTransform:: unable to execute on strings. Please write logic "
        "higher level in each op for the string data type.";

    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (!DataTypeUtils::isZ(zType)) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execPairwiseIntTransform requires Z operand to have INT type";
    errorMessage += "X type: ";
    errorMessage += DataTypeUtils::asString(xType);
    errorMessage += "Y type: ";
    errorMessage += DataTypeUtils::asString(yType);
    errorMessage += "Z type: ";
    errorMessage += DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (yType != xType || zType != xType) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execPairwiseIntTransform both operands must have same data type x type:";
    errorMessage += DataTypeUtils::asString(xType);
    errorMessage += " y type: ";
    errorMessage += DataTypeUtils::asString(yType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  dim3 launchDims = getLaunchDims("pairwiseTransforms");

  BUILD_SINGLE_SELECTOR(
      xType, functions::pairwise_transforms::PairWiseIntTransform,
      ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraParams),
      SD_INTEGER_TYPES)
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execSummaryStatsScalar(LaunchContext* lc, int opNum, void const* hX,
                                                 LongType const* hXShapeInfo, void const* dX,
                                                 LongType const* dXShapeInfo, void* extraParams, void* hZ,
                                                 LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo,
                                                 bool biasCorrected) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  dim3 launchDims = getLaunchDims("summaryStats");

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execSummaryStatsScalar:: unable to execute on strings. Please write logic higher level "
        "in each op for the string data type.")
  }
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::summarystats::SummaryStatsReduce,
      ::execSummaryStatsReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ,
                                     dZShapeInfo, hZShapeInfo, nullptr, nullptr, biasCorrected, reductionPointer),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcastBool(LaunchContext* lc, int opNum, void const* hX, LongType const* hXShapeInfo,
                                            void const* dX, LongType const* dXShapeInfo, void const* hY,
                                            LongType const* hYShapeInfo, void const* dY, LongType const* dYShapeInfo,
                                            void* hZ, LongType const* hZShapeInfo, void* dZ,
                                            LongType const* dZShapeInfo, void* extraParams, LongType* dimension,
                                            LongType dimensionLength, LongType const* tadOnlyShapeInfo,
                                            LongType const* tadOffsets, LongType const* tadOnlyShapeInfoZ,
                                            LongType const* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hYShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execBroadcastBool:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }

  if (!DataTypeUtils::isB(zType))
    THROW_EXCEPTION("NativeOpExecutioner::execBroadcastBool requires Z operand to have BOOL type");

  if (yType != xType)
    THROW_EXCEPTION("NativeOpExecutioner::execBroadcastBool requires both X & Y operands to have same type");

  if (Environment::getInstance().isDebugAndVerbose()) printf("F3B opType:[%i]\n", opNum);

  dim3 launchDims = getLaunchDims("broadcast");

  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::broadcast::BroadcastBool,
      ::execBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraParams,
                      dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
      SD_COMMON_TYPES, SD_BOOL_TYPES)
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcastBool(LaunchContext* lc, const int opNum, const void* hX,
                                            const LongType* hXShapeInfo, const void* dX, const LongType* dXShapeInfo,
                                            const void* hY, const LongType* hYShapeInfo, const void* dY,
                                            const LongType* dYShapeInfo, void* hZ, const LongType* hZShapeInfo,
                                            void* dZ, const LongType* dZShapeInfo, void* extraParams) {
  auto stream = lc->getCudaStream();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
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
    LaunchContext* lc, int opNum, void const* hX, LongType const* hXShapeInfo, void const* dX,
    LongType const* dXShapeInfo, void const* hY, LongType const* hYShapeInfo, void const* dY,
    LongType const* dYShapeInfo, void* hZ, LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo,
    void* extraParams, LongType* dimension, LongType dimensionLength, LongType const* tadOnlyShapeInfo,
    LongType const* tadOffsets, LongType const* tadOnlyShapeInfoZ, LongType const* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hYShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execInverseBroadcastBool:: unable to execute on strings. Please write logic higher level "
        "in each op for the string data type.")
  }
  if (!DataTypeUtils::isB(zType))
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
void NativeOpExecutioner::execBroadcastInt(LaunchContext* lc, int opNum, void const* hX,
                                           LongType const* hXShapeInfo, void const* dX, LongType const* dXShapeInfo,
                                           void const* hY, LongType const* hYShapeInfo, void const* dY,
                                           LongType const* dYShapeInfo, void* hZ, LongType const* hZShapeInfo, void* dZ,
                                           LongType const* dZShapeInfo, LongType* dimension, LongType dimensionLength,
                                           LongType const* tadOnlyShapeInfo, LongType const* tadOffsets,
                                           LongType const* tadOnlyShapeInfoZ, LongType const* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hYShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execBroadcastInt:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  if (!DataTypeUtils::isZ(zType))
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
void NativeOpExecutioner::execBroadcastInt(LaunchContext* lc, const int opNum, const void* hX,
                                           const LongType* hXShapeInfo, const void* dX, const LongType* dXShapeInfo,
                                           const void* hY, const LongType* hYShapeInfo, const void* dY,
                                           const LongType* dYShapeInfo, void* hZ, const LongType* hZShapeInfo, void* dZ,
                                           const LongType* dZShapeInfo) {
  auto stream = lc->getCudaStream();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hYShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOPExecutioner::execBroadcastInt:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }

  if (!DataTypeUtils::isZ(zType))
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
    LaunchContext* lc, int opNum, void const* hX, LongType const* hXShapeInfo, void const* dX,
    LongType const* dXShapeInfo, void const* hY, LongType const* hYShapeInfo, void const* dY,
    LongType const* dYShapeInfo, void* hZ, LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo,
    LongType* dimension, LongType dimensionLength, LongType const* tadOnlyShapeInfo, LongType const* tadOffsets,
    LongType const* tadOnlyShapeInfoZ, LongType const* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hYShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);

  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execInverseBroadcastInt:: unable to execute on strings. Please write logic higher level "
        "in each op for the string data type.")
  }
  if (!DataTypeUtils::isZ(zType))
    THROW_EXCEPTION("NativeOpExecutioner::execInverseBroadcastInt requires Z operand to have INT type");

  if (yType != xType || zType != xType)
    THROW_EXCEPTION("NativeOpExecutioner::execInverseBroadcastInt requires both X & Y operands to have same type");

  if (Environment::getInstance().isDebugAndVerbose()) printf("F3BI opType:[%i]\n", opNum);

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
void NativeOpExecutioner::execBroadcast(LaunchContext* lc, int opNum, void const* hX, LongType const* hXShapeInfo,
                                        void const* dX, LongType const* dXShapeInfo, void const* hY,
                                        LongType const* hYShapeInfo, void const* dY, LongType const* dYShapeInfo,
                                        void* hZ, LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo,
                                        LongType* dimension, LongType dimensionLength, LongType const* tadOnlyShapeInfo,
                                        LongType const* tadOffsets, LongType const* tadOnlyShapeInfoZ,
                                        LongType const* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hYShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);

  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOPExecutioner::execBroadcast:: unable to execute on strings. Please write logic higher level in each op "
        "for the string data type.")
  }
  dim3 launchDims = getLaunchDims("broadcast");

#ifdef SD_EXPERIMENTAL_ENABLED
  BUILD_PAIRWISE_SELECTOR(
      xType, yType, zType, functions::broadcast::Broadcast,
      ::execBroadcast(launchDims, stream, opType, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, dimension,
                      dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
      SD_COMMON_TYPES, SD_COMMON_TYPES);
#else
  BUILD_SINGLE_SELECTOR_THRICE(
      xType, functions::broadcast::Broadcast,
      ::execBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, dimension,
                      dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
      SD_COMMON_TYPES);
#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execBroadcast(LaunchContext* lc, const int opNum, const void* hX,
                                        const LongType* hXShapeInfo, const void* dX, const LongType* dXShapeInfo,
                                        const void* hY, const LongType* hYShapeInfo, const void* dY,
                                        const LongType* dYShapeInfo, void* hZ, const LongType* hZShapeInfo, void* dZ,
                                        const LongType* dZShapeInfo) {
  auto stream = lc->getCudaStream();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hYShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execBroadcast:: unable to execute on strings. Please write logic higher level in each op "
        "for the string data type.")
  }

  dim3 launchDims = getLaunchDims("broadcast");
  // shared memory

#ifdef SD_EXPERIMENTAL_ENABLED
  BUILD_PAIRWISE_SELECTOR(
      xType, yType, zType, functions::broadcast::Broadcast,
      ::execBroadcast(launchDims, stream, opType, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo), SD_COMMON_TYPES,
      SD_COMMON_TYPES);
#else
  BUILD_SINGLE_SELECTOR_THRICE(
      xType, functions::broadcast::Broadcast,
      ::execBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo), SD_COMMON_TYPES);
#endif
}

void NativeOpExecutioner::execInverseBroadcast(LaunchContext* lc, int opNum, void const* hX,
                                               LongType const* hXShapeInfo, void const* dX, LongType const* dXShapeInfo,
                                               void const* hY, LongType const* hYShapeInfo, void const* dY,
                                               LongType const* dYShapeInfo, void* hZ, LongType const* hZShapeInfo,
                                               void* dZ, LongType const* dZShapeInfo, LongType* dimension,
                                               LongType dimensionLength, LongType const* tadOnlyShapeInfo,
                                               LongType const* tadOffsets, LongType const* tadOnlyShapeInfoZ,
                                               LongType const* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hYShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execInverseBroadcast:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }

  dim3 launchDims = getLaunchDims("broadcast");

#ifdef SD_EXPERIMENTAL_ENABLED
  BUILD_PAIRWISE_SELECTOR(
      xType, yType, zType, functions::broadcast::Broadcast,
      ::execInverseBroadcast(launchDims, stream, opType, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, dimension,
                             dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
      SD_COMMON_TYPES, SD_COMMON_TYPES);
#else
  BUILD_SINGLE_SELECTOR_THRICE(
      xType, functions::broadcast::Broadcast,
      ::execInverseBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, dimension,
                             dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
      SD_COMMON_TYPES);
#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceSame(LaunchContext* lc, int opNum, void const* hX, LongType const* hXShapeInfo,
                                         void const* dX, LongType const* dXShapeInfo, void* extraParams, void* hZ,
                                         LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo,
                                         LongType* dimension, LongType dimensionLength) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  if (Environment::getInstance().isDebugAndVerbose()) printf("SF7 opType:[%i]\n", opNum);

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduceSame:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  if (zType != xType)
    throw datatype_exception::build(
        "NativeOpExecutioner::execReduceSame requires both X & Z operands to have same type", xType, zType);

  auto numBlocks = shape::length(hZShapeInfo);
  dim3 launchDims = getReduceDims(numBlocks);

  BUILD_SINGLE_SELECTOR(xType, functions::reduce::ReduceSameFunction,
                        ::execReduceXD(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams,
                                       reductionPointer, dZ, dZShapeInfo, hZShapeInfo, dimension),
                        SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceLong(LaunchContext* lc, int opNum, void const* hX, LongType const* hXShapeInfo,
                                         void const* dX, LongType const* dXShapeInfo, void* extraParams, void* hZ,
                                         LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo,
                                         LongType* dimension, LongType dimensionLength) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  if (Environment::getInstance().isDebugAndVerbose()) printf("LF7 opType:[%i]\n", opNum);

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduceLong:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  if (zType != INT64) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execReduceLong requires Z operand to have INT64 type";
    errorMessage += "X type: ";
    errorMessage += DataTypeUtils::asString(xType);
    errorMessage += "Y type: ";
    errorMessage += DataTypeUtils::asString(zType);
    errorMessage += "Z type: ";
    errorMessage += DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  auto numBlocks = shape::length(hZShapeInfo);
  dim3 launchDims = getReduceDims(numBlocks);

  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceLongFunction,
                        ::execReduceXD(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams,
                                       reductionPointer, dZ, dZShapeInfo, hZShapeInfo, dimension),
                        SD_COMMON_TYPES, SD_LONG_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceBool(LaunchContext* lc, int opNum, void const* hX, LongType const* hXShapeInfo,
                                         void const* dX, LongType const* dXShapeInfo, void* extraParams, void* hZ,
                                         LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo,
                                         LongType* dimension, LongType dimensionLength) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  if (Environment::getInstance().isDebugAndVerbose()) printf("BF7 opType:[%i]\n", opNum);

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduceBool:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  if (zType != BOOL)
    THROW_EXCEPTION("NativeOpExecutioner::execReduceBool requires Z operand to have BOOL type");

  auto numBlocks = shape::length(hZShapeInfo);
  dim3 launchDims = getReduceDims(numBlocks);

  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceBoolFunction,
                        ::execReduceXD(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams,
                                       reductionPointer, dZ, dZShapeInfo, hZShapeInfo, dimension),
                        SD_COMMON_TYPES, SD_BOOL_TYPES);
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param extraParams
 * @param dZ
 * @param dZShapeInfo
 */
void NativeOpExecutioner::execReduceFloat(LaunchContext* lc, int opNum, const void* hX, const LongType* hXShapeInfo,
                                          const void* dX, const LongType* dXShapeInfo, void* extraParams, void* hZ,
                                          const LongType* hZShapeInfo, void* dZ, const LongType* dZShapeInfo,
                                          LongType* dimension, LongType dimensionLength) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  if (Environment::getInstance().isDebugAndVerbose()) printf("F8 opType:[%i]\n", opNum);

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduceFloat:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  auto numBlocks = shape::length(hZShapeInfo);
  dim3 launchDims = getReduceDims(numBlocks);

  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceFloatFunction,
                        ::execReduceXD(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams,
                                       reductionPointer, dZ, dZShapeInfo, hZShapeInfo, dimension),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param extraParams
 * @param dZ
 * @param dZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void NativeOpExecutioner::execIndexReduce(LaunchContext* lc, int opNum, void const* hX, LongType const* hXShapeInfo,
                                          void const* dX, LongType const* dXShapeInfo, void* extraParams, void* hZ,
                                          LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo,
                                          LongType* dimension, LongType dimensionLength, LongType const* tadShapeInfo,
                                          LongType const* tadOffsets) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();
  auto allocationPointer = lc->getAllocationPointer();

  if (Environment::getInstance().isDebugAndVerbose()) {
    printf("F2 opType:[%i]\n", opNum);
  }
  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execIndexReduce:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  auto numBlocks = shape::length(hZShapeInfo);
  auto tadLength = shape::length(hXShapeInfo) / numBlocks;
  dim3 launchDims = getReduceDims(numBlocks);
  if (zType != INT64 && zType != INT32)
    throw datatype_exception::build("NativeOpExecutioner::execIndexReduce requires Z operand to have INT32/INT64 type",
                                    zType);

  auto dz = reinterpret_cast<LongType*>(dZ);

  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::indexreduce::IndexReduce,
      ::executeIndexReduce(launchDims,
                           stream,
                           opNum,
                           dX,
                           dXShapeInfo, shape::rank(hXShapeInfo),
                           extraParams,
                           dz,
                           dZShapeInfo,
                           shape::rank(hZShapeInfo),
                           dimension,
                           dimensionLength,
                           1,
                           allocationPointer,
                           reductionPointer,
                           tadShapeInfo,
                           tadOffsets),
      SD_COMMON_TYPES, SD_INDEXING_TYPES);
}

/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param extraParams
 */
////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execIndexReduceScalar(LaunchContext* lc, int opNum, void const* hX,
                                                LongType const* hXShapeInfo, void const* dX,
                                                LongType const* dXShapeInfo, void* extraParams, void* hZ,
                                                LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();
  LongType* allocationPointer = lc->getAllocationPointer();

  auto xLength = shape::length(hXShapeInfo);
  dim3 launchDims = getReduceDims(xLength);

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execIndexReduceScalar:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }

  if (zType != INT64 && zType != INT32)
    throw datatype_exception::build(
        "NativeOpExecutioner::execIndexReduceScalar requires Z operand to have INT32/INT64 data type", zType);

  auto dz = reinterpret_cast<LongType*>(dZ);

  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::indexreduce::IndexReduce,
      ::executeIndexReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, shape::rank(hXShapeInfo), extraParams, dz,
                                 dZShapeInfo, 0, nullptr, 0, 1, allocationPointer, reductionPointer, nullptr, nullptr),
      SD_COMMON_TYPES, SD_INDEXING_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceFloatScalar(LaunchContext* lc, int opNum, void const* hX,
                                                LongType const* hXShapeInfo, void const* dX,
                                                LongType const* dXShapeInfo, void* extraParams, void* hZ,
                                                LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduceFloatScalar:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }
  auto xLength = shape::length(hXShapeInfo);
  dim3 launchDims = getReduceDims(xLength);

  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceFloatFunction,
                        ::execReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ,
                                           dZShapeInfo, hZShapeInfo, nullptr, 0, reductionPointer, nullptr),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceBoolScalar(LaunchContext* lc, int opNum, void const* hX,
                                               LongType const* hXShapeInfo, void const* dX, LongType const* dXShapeInfo,
                                               void* extraParams, void* hZ, LongType const* hZShapeInfo, void* dZ,
                                               LongType const* dZShapeInfo) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduceBoolScalar:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }
  if (zType != BOOL)
    THROW_EXCEPTION("NativeOpExecutioner::execReduceBoolScalar requires Z operand to have BOOL type");

  auto xLength = shape::length(hXShapeInfo);
  dim3 launchDims = getReduceDims(xLength);

  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceBoolFunction,
                        ::execReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ,
                                           dZShapeInfo, hZShapeInfo, nullptr, 0, reductionPointer, nullptr),
                        SD_COMMON_TYPES, SD_BOOL_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceSameScalar(LaunchContext* lc, int opNum, void const* hX,
                                               LongType const* hXShapeInfo, void const* dX, LongType const* dXShapeInfo,
                                               void* extraParams, void* hZ, LongType const* hZShapeInfo, void* dZ,
                                               LongType const* dZShapeInfo) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduceSameScalar:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }
  if (zType != xType)
    throw datatype_exception::build(
        "NativeOpExecutioner::execReduceSameScalar requires both X & Z operands to have same type", xType, zType);

  auto xLength = shape::length(hXShapeInfo);
  dim3 launchDims = getReduceDims(xLength);

  BUILD_SINGLE_SELECTOR(xType, functions::reduce::ReduceSameFunction,
                        ::execReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ,
                                           dZShapeInfo, hZShapeInfo, nullptr, 0, reductionPointer, nullptr),
                        SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceLongScalar(LaunchContext* lc, int opNum, void const* hX,
                                               LongType const* hXShapeInfo, void const* dX, LongType const* dXShapeInfo,
                                               void* extraParams, void* hZ, LongType const* hZShapeInfo, void* dZ,
                                               LongType const* dZShapeInfo) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduceLongScalar:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }
  if (zType != INT64)
    throw datatype_exception::build("NativeOpExecutioner::execReduceLongScalar wrong Z data type", INT64,
                                    zType);

  auto xLength = shape::length(hXShapeInfo);
  dim3 launchDims = getReduceDims(xLength);

  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceLongFunction,
                        ::execReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ,
                                           dZShapeInfo, hZShapeInfo, nullptr, 0, reductionPointer, nullptr),
                        SD_COMMON_TYPES, SD_LONG_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformSame(LaunchContext* lc, int opNum, void const* hX,
                                            LongType const* hXShapeInfo, void const* dX, LongType const* dXShapeInfo,
                                            void* hZ, LongType const* hZShapeInfo, void* dZ,
                                            LongType const* dZShapeInfo, void* extraParams,
                                            LongType const* tadShapeInfo, LongType const* tadOffsets) {
  auto stream = lc->getCudaStream();

  auto xRank = shape::rank(hXShapeInfo);
  auto zRank = shape::rank(hZShapeInfo);
  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
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
                                                 dZShapeInfo, zRank, nullptr, nullptr, nullptr, nullptr),
                        SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformBool(LaunchContext* lc, int opNum, void const* hX,
                                            LongType const* hXShapeInfo, void const* dX, LongType const* dXShapeInfo,
                                            void* hZ, LongType const* hZShapeInfo, void* dZ,
                                            LongType const* dZShapeInfo, void* extraParams,
                                            LongType const* tadShapeInfo, LongType const* tadOffsets) {
  auto stream = lc->getCudaStream();

  auto xRank = shape::rank(hXShapeInfo);
  auto zRank = shape::rank(hZShapeInfo);
  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execTransformBool:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }
  if (!DataTypeUtils::isB(zType)) {
    THROW_EXCEPTION("NativeOpExecutioner::execTransformBool requires Z to have same boolean type");
  }

  dim3 launchDims = getLaunchDims("transformScan");
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformBool,
                        ::executeTransformShaped(launchDims, stream, opNum, dX, dXShapeInfo, xRank, extraParams, dZ,
                                                 dZShapeInfo, zRank, nullptr, nullptr, nullptr, nullptr),
                        SD_COMMON_TYPES, SD_BOOL_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformAny(LaunchContext* lc, int opNum, void const* hX,
                                           LongType const* hXShapeInfo, void const* dX, LongType const* dXShapeInfo,
                                           void* hZ, LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo,
                                           void* extraParams, LongType const* tadShapeInfo, LongType const* tadOffsets,
                                           bool allowParallelism) {
  auto stream = lc->getCudaStream();

  auto xRank = shape::rank(hXShapeInfo);
  auto zRank = shape::rank(hZShapeInfo);
  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);

  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execTransformAny:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  dim3 launchDims = getLaunchDims("transformScan");
  if (DataTypeUtils::isS(xType)) {
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformAny,
                          ::executeTransformShaped(launchDims, stream, opNum, dX, dXShapeInfo, xRank, extraParams, dZ,
                                                   dZShapeInfo, zRank, nullptr, nullptr, nullptr, nullptr),
                          SD_STRING_TYPES, SD_STRING_TYPES);
  } else {
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformAny,
                          ::executeTransformShaped(launchDims, stream, opNum, dX, dXShapeInfo, xRank, extraParams, dZ,
                                                   dZShapeInfo, zRank, nullptr, nullptr, nullptr, nullptr),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);
  }
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformStrict(LaunchContext* lc, int opNum, void const* hX,
                                              LongType const* hXShapeInfo, void const* dX, LongType const* dXShapeInfo,
                                              void* hZ, LongType const* hZShapeInfo, void* dZ,
                                              LongType const* dZShapeInfo, void* extraParams,
                                              LongType const* tadShapeInfo, LongType const* tadOffsets) {
  auto stream = lc->getCudaStream();

  auto xRank = shape::rank(hXShapeInfo);
  auto zRank = shape::rank(hZShapeInfo);
  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);

  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execTransformStrict:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }

  if (xType != zType || !DataTypeUtils::isR(xType)) {
    throw datatype_exception::build(
        "NativeOpExecutioner::execTransformStrict requires X & Z to have same floating point type", xType, zType);
  }

  dim3 launchDims = getLaunchDims("transformScan");
  BUILD_SINGLE_SELECTOR(xType, functions::transform::TransformStrict,
                        ::executeTransformShaped(launchDims, stream, opNum, dX, dXShapeInfo, xRank, extraParams, dZ,
                                                 dZShapeInfo, zRank, nullptr, nullptr, nullptr, nullptr),
                        SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execTransformFloat(LaunchContext* lc, int opNum, void const* hX,
                                             LongType const* hXShapeInfo, void const* dX, LongType const* dXShapeInfo,
                                             void* hZ, LongType const* hZShapeInfo, void* dZ,
                                             LongType const* dZShapeInfo, void* extraParams,
                                             LongType const* tadShapeInfo, LongType const* tadOffsets) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  auto xRank = shape::rank(hXShapeInfo);
  auto zRank = shape::rank(hZShapeInfo);
  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execTransformFloat:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }

  if (!DataTypeUtils::isR(zType))
    throw datatype_exception::build("NativeOpExecutioner::execTransformFloat requires Z to have floating point type",
                                    zType);

  dim3 launchDims = getLaunchDims("transformScan");
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformFloat,
                        ::executeTransformShaped(launchDims, stream, opNum, dX, dXShapeInfo, xRank, extraParams, dZ,
                                                 dZShapeInfo, zRank, nullptr, nullptr, nullptr, nullptr),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);

  fflush(stdout);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execSummaryStats(LaunchContext* lc, int opNum, void const* hX,
                                           LongType const* hXShapeInfo, void const* dX, LongType const* dXShapeInfo,
                                           void* extraParams, void* hZ, LongType const* hZShapeInfo, void* dZ,
                                           LongType const* dZShapeInfo, bool biasCorrected) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  dim3 launchDims = getLaunchDims("summaryStats");

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execSummaryStats:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  if (!DataTypeUtils::isR(zType))
    throw datatype_exception::build(
        "NativeOpExecutioner::execSummaryStats requires Z operand to have floating point data type", zType);

  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::summarystats::SummaryStatsReduce,
      ::execSummaryStatsReduce(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ, dZShapeInfo,
                               hZShapeInfo, nullptr, nullptr, biasCorrected, reductionPointer),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execSummaryStats(LaunchContext* lc, int opNum, void const* hX,
                                           LongType const* hXShapeInfo, void const* dX, LongType const* dXShapeInfo,
                                           void* extraParams, void* hZ, LongType const* hZShapeInfo, void* dZ,
                                           LongType const* dZShapeInfo, LongType* dimension, LongType dimensionLength,
                                           LongType const* tadShapeInfo, LongType const* tadOffsets,
                                           bool biasCorrected) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  dim3 launchDims = getLaunchDims("summaryStats");

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execSummaryStats:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  if (!DataTypeUtils::isR(zType))
    throw datatype_exception::build(
        "NativeOpExecutioner::execSummaryStats requires Z operand to have floating point data type", zType);

  BUILD_DOUBLE_SELECTOR(xType, zType, functions::summarystats::SummaryStatsReduce,
                        ::execSummaryStatsReduce(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams,
                                                 dZ, dZShapeInfo, hZShapeInfo, dimension, dimensionLength, tadShapeInfo,
                                                 tadOffsets, biasCorrected, reductionPointer),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3(LaunchContext* lc, int opNum, void const* hX, LongType const* hXShapeInfo,
                                      void const* dX, LongType const* dXShapeInfo, void* extraParams, void const* hY,
                                      LongType const* hYShapeInfo, void const* dY, LongType const* dYShapeInfo,
                                      void* hZ, LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();
  auto allocationPointer = lc->getAllocationPointer();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hYShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduce3:: unable to execute on strings. Please write logic higher level in each op "
        "for the string data type.")
  }

  dim3 launchDims = getReduceDims(shape::length(hXShapeInfo));

  if (xType != yType)
    throw datatype_exception::build("NativeOpExecutioner::execReduce3 requires Y operand to have X type", xType,
                                        yType);

  if (!DataTypeUtils::isR(zType))
    throw datatype_exception::build(
        "NativeOpExecutioner::execReduce3 requires Z operand to have floating point data type", zType);

  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3,
                        ::execScalar(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParams, dZ,
                                     dZShapeInfo, allocationPointer, reductionPointer, nullptr),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3(LaunchContext* lc, int opNum, const void* hX, const LongType* hXShapeInfo,
                                      const void* dX, const LongType* dXShapeInfo, void* extraParamsVals,
                                      const void* hY, const LongType* hYShapeInfo, const void* dY,
                                      const LongType* dYShapeInfo, void* hZ, const LongType* hZShapeInfo, void* dZ,
                                      const LongType* dZShapeInfo, LongType* dimension, LongType dimensionLength,
                                      const LongType* xTadOnlyShapeInfo, const LongType* xTadOffsets,
                                      const LongType* yTadOnlyShapeInfo, const LongType* yTadOffsets) {
  if (shape::isScalar(hZShapeInfo)) {
    execReduce3(lc, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParamsVals, hY, hYShapeInfo, dY,
                                     dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo);
    return;
  }

  auto stream = lc->getCudaStream();
  auto allocationPointer = lc->getAllocationPointer();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hYShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduce3:: unable to execute on strings. Please write logic higher level in each op "
        "for the string data type.")
  }
  if (xType != yType)
    throw datatype_exception::build("NativeOpExecutioner::execReduce3 requires Y operand to have X type", xType,
                                        yType);

  if (!DataTypeUtils::isR(zType))
    throw datatype_exception::build(
        "NativeOpExecutioner::execReduce3 requires Z operand to have floating point data type", zType);

  auto numBlocks = shape::length(hZShapeInfo);
  dim3 launchDims = getReduceDims(numBlocks);

  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::reduce3::Reduce3,
      ::exec(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParamsVals, dZ, dZShapeInfo, dimension,
             dimensionLength, 1, allocationPointer, xTadOnlyShapeInfo, xTadOffsets, yTadOnlyShapeInfo, yTadOffsets),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3Scalar(LaunchContext* lc, int opNum, void const* hX,
                                            LongType const* hXShapeInfo, void const* dX, LongType const* dXShapeInfo,
                                            void* extraParams, void const* hY, LongType const* hYShapeInfo,
                                            void const* dY, LongType const* dYShapeInfo, void* hZ,
                                            LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo) {
  auto stream = lc->getCudaStream();
  auto allocationPointer = lc->getAllocationPointer();
  auto reductionPointer = lc->getReductionPointer();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hYShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduce3Scalar:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }

  dim3 launchDims = getReduceDims(shape::length(hXShapeInfo));

  if (xType != yType)
    throw datatype_exception::build("NativeOpExecutioner::execReduce3Scalar requires Y operand to have X type",
                                        xType, yType);

  if (!DataTypeUtils::isR(zType))
    throw datatype_exception::build(
        "NativeOpExecutioner::execReduce3Scalar requires Z operand to have floating point data type", zType);

  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3,
                        ::execScalar(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParams, dZ,
                                     dZShapeInfo, allocationPointer, reductionPointer, nullptr),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarBool(LaunchContext* lc, int opNum, void const* hX, LongType const* hXShapeInfo,
                                         void const* dX, LongType const* dXShapeInfo, void* hZ,
                                         LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo,
                                         void const* hScalar, LongType const* hScalarShapeInfo, void const* dScalar,
                                         LongType const* dScalarShapeInfo, void* extraParams, bool allowParallelism) {
  auto stream = lc->getCudaStream();

  dim3 launchDims = getLaunchDims("scalarScan");

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execScalarBool:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  if (xType != yType) THROW_EXCEPTION("NativeOpExecutioner::execScalarBool requires X & Y to have same type");

  if (!DataTypeUtils::isB(zType))
    THROW_EXCEPTION("NativeOpExecutioner::execScalarBool requires Z operand to have BOOL type");

  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::scalar::ScalarBoolTransform,
      ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalar, extraParams),
      SD_COMMON_TYPES, SD_BOOL_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarBool(LaunchContext* lc, int opNum, const void* hX, const LongType* hXShapeInfo,
                                         const void* dX, const LongType* dXShapeInfo, void* extraParams, void* hZ,
                                         const LongType* hZShapeInfo, void* dZ, const LongType* dZShapeInfo,
                                         const void* hScalars, const LongType* hScalarShapeInfo, const void* dScalars,
                                         const LongType* dScalarShapeInfo, LongType* dimension,
                                         LongType dimensionLength, const LongType* tadShapeInfo,
                                         const LongType* tadOffsets, const LongType* tadShapeInfoZ,
                                         const LongType* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  dim3 launchDims = getLaunchDims("scalarScan");

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execScalarBool:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }

  if (xType != yType) THROW_EXCEPTION("NativeOpExecutioner::execScalarBool requires X & Y to have same type");

  if (!DataTypeUtils::isB(zType))
    THROW_EXCEPTION("NativeOpExecutioner::execScalarBool requires Z operand to have BOOL type");

  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::scalar::ScalarBoolTransform,
      ::executeCudaAlongDimension(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalars, extraParams,
                                  dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ),
      SD_COMMON_TYPES, SD_BOOL_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarInt(LaunchContext* lc, int opNum, void const* hX, LongType const* hXShapeInfo,
                                        void const* dX, LongType const* dXShapeInfo, void* hZ,
                                        LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo,
                                        void const* hScalar, LongType const* hScalarShapeInfo, void const* dScalar,
                                        LongType const* dScalarShapeInfo, void* extraParams, bool allowParallelism) {
  auto stream = lc->getCudaStream();

  dim3 launchDims = getLaunchDims("scalarScan");

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);
  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execScalarInt:: unable to execute on strings. Please write logic higher level in each op "
        "for the string data type.")
  }

  if (xType != yType || zType != xType)
    THROW_EXCEPTION("NativeOpExecutioner::execScalarInt requires X & Y to have same type");

  if (!DataTypeUtils::isZ(zType))
    THROW_EXCEPTION("NativeOpExecutioner::execScalarInt requires Z operand to have INT type");

  BUILD_SINGLE_SELECTOR(
      xType, functions::scalar::ScalarIntTransform,
      ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalar, extraParams),
      SD_INTEGER_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarInt(LaunchContext* lc, int opNum, const void* hX, const LongType* hXShapeInfo,
                                        const void* dX, const LongType* dXShapeInfo, void* extraParams, void* hZ,
                                        const LongType* hZShapeInfo, void* dZ, const LongType* dZShapeInfo,
                                        const void* hScalars, const LongType* hScalarShapeInfo, const void* dScalars,
                                        const LongType* dScalarShapeInfo, LongType* dimension, LongType dimensionLength,
                                        const LongType* tadShapeInfo, const LongType* tadOffsets,
                                        const LongType* tadShapeInfoZ, const LongType* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  dim3 launchDims = getLaunchDims("scalarScan");

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);

  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execScalarInt:: unable to execute on strings. Please write logic higher level in each op "
        "for the string data type.")
  }
  if (xType != yType || zType != xType)
    THROW_EXCEPTION("NativeOpExecutioner::execScalarInt requires X & Y to have same type");

  if (!DataTypeUtils::isZ(zType))
    THROW_EXCEPTION("NativeOpExecutioner::execScalarInt requires Z operand to have INT type");

  BUILD_SINGLE_SELECTOR(
      xType, functions::scalar::ScalarIntTransform,
      ::executeCudaAlongDimension(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalars, extraParams,
                                  dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ),
      SD_INTEGER_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalar(LaunchContext* lc, int opNum, void const* hX, LongType const* hXShapeInfo,
                                     void const* dX, LongType const* dXShapeInfo, void* hZ, LongType const* hZShapeInfo,
                                     void* dZ, LongType const* dZShapeInfo, void const* hScalar,
                                     LongType const* hScalarShapeInfo, void const* dScalar,
                                     LongType const* dScalarShapeInfo, void* extraParams, bool allowParallelism) {
  auto stream = lc->getCudaStream();

  dim3 launchDims = getLaunchDims("scalarScan");

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);

  if (DataTypeUtils::isS(xType) || DataTypeUtils::isS(yType) || DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execScalar:: unable to execute on strings. Please write logic higher level in each op "
        "for the string data type.")
  }
  BUILD_SINGLE_SELECTOR_THRICE(xType, functions::scalar::ScalarTransform,
                               ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, dZ,
                                                   dZShapeInfo, hZShapeInfo, dScalar, extraParams),
                               SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalar(LaunchContext* lc, int opNum, void const* hX, LongType const* hXShapeInfo,
                                     void const* dX, LongType const* dXShapeInfo, void* extraParams, void* hZ,
                                     LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo,
                                     void const* hScalars, LongType const* hScalarShapeInfo, void const* dScalars,
                                     LongType const* dScalarShapeInfo, LongType* dimension, LongType dimensionLength,
                                     LongType const* tadShapeInfo, LongType const* tadOffsets,
                                     LongType const* tadShapeInfoZ, LongType const* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);

  dim3 launchDims = getLaunchDims("scalarScan");

#ifdef SD_EXPERIMENTAL_ENABLED
  BUILD_PAIRWISE_SELECTOR(
      xType, yType, zType, functions::scalar::ScalarTransform,
      ::executeCudaAlongDimension(launchDims, stream, opType, dX, dXShapeInfo, dZ, dZShapeInfo, dScalars, extraParams,
                                  dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ),
      SD_COMMON_TYPES, SD_COMMON_TYPES);
#else

  if (xType != yType || xType != zType) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execScalar requires both X & Y to have same data type";
    errorMessage += "X data type: ";
    errorMessage += DataTypeUtils::asString(xType);
    errorMessage += ", Y data type: ";
    errorMessage += DataTypeUtils::asString(yType);
    errorMessage += ", Z data type: ";
    errorMessage += DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (DataTypeUtils::isS(xType)) {
    BUILD_SINGLE_SELECTOR_THRICE(
        xType, functions::scalar::ScalarTransform,
        ::executeCudaAlongDimension(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalars, extraParams,
                                    dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ),
        SD_STRING_TYPES);
  } else {
    BUILD_SINGLE_SELECTOR_THRICE(
        xType, functions::scalar::ScalarTransform,
        ::executeCudaAlongDimension(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalars, extraParams,
                                    dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ),
        SD_COMMON_TYPES);
  }

#endif

  // TODO: remove after the release
  auto res = cudaStreamSynchronize(*stream);
  if (res != 0) throw cuda_exception::build("execScalar B failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(LaunchContext* lc, int opNum, Pointer stateHost, void* hZ,
                                     LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo,
                                     void* extraArguments) {
  auto stream = lc->getCudaStream();
  auto sizeOf = sizeof(graph::RandomGenerator);
  Pointer stateDevice;

  cudaError_t res = cudaMalloc(reinterpret_cast<void**>(&stateDevice), sizeOf);
  checkCudaErrors(cudaStreamSynchronize(*stream));
  checkCudaErrors(cudaMemcpyAsync(stateDevice, stateHost, sizeOf, cudaMemcpyHostToDevice, *stream));

  dim3 launchDims = getLaunchDims("random");
  auto zType = ArrayOptions::dataType(hZShapeInfo);

  auto rng = reinterpret_cast<graph::RandomGenerator*>(stateHost);

  BUILD_SINGLE_SELECTOR(zType, functions::random::RandomFunction,
                        ::executeCudaSingle(launchDims, stream, opNum, stateDevice, dZ, dZShapeInfo, extraArguments),
                        SD_FLOAT_TYPES);

  res = cudaStreamSynchronize(*stream);
  if (res != 0) throw cuda_exception::build("execRandom X failed", res);

  cudaFree(stateDevice);

  rng->rewindH(shape::length(hZShapeInfo));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(LaunchContext* lc, int opNum, Pointer stateHost, void const* hX,
                                     LongType const* hXShapeInfo, void const* dX, LongType const* dXShapeInfo, void* hZ,
                                     LongType const* hZShapeInfo, void* dZ, LongType const* dZShapeInfo,
                                     void* extraArguments) {
  auto stream = lc->getCudaStream();

  auto sizeOf = sizeof(graph::RandomGenerator);
  Pointer stateDevice;

  cudaError_t res = cudaMalloc(reinterpret_cast<void**>(&stateDevice), sizeOf);
  checkCudaErrors(cudaStreamSynchronize(*stream));
  checkCudaErrors(cudaMemcpyAsync(stateDevice, stateHost, sizeOf, cudaMemcpyHostToDevice, *stream));

  auto rng = reinterpret_cast<graph::RandomGenerator*>(stateHost);

  dim3 launchDims = getLaunchDims("random");
  auto xType = ArrayOptions::dataType(hZShapeInfo);

  BUILD_SINGLE_SELECTOR(
      xType, functions::random::RandomFunction,
      ::executeCudaDouble(launchDims, stream, opNum, stateDevice, dX, dXShapeInfo, dZ, dZShapeInfo, extraArguments),
      SD_FLOAT_TYPES);

  res = cudaStreamSynchronize(*stream);
  if (res != 0) throw cuda_exception::build("execRandom XY failed", res);

  cudaFree(stateDevice);

  rng->rewindH(shape::length(hZShapeInfo));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(LaunchContext* lc, int opNum, Pointer stateHost, void const* hX,
                                     LongType const* hXShapeInfo, void const* dX, LongType const* dXShapeInfo,
                                     void const* hY, LongType const* hYShapeInfo, void const* dY,
                                     LongType const* dYShapeInfo, void* hZ, LongType const* hZShapeInfo, void* dZ,
                                     LongType const* dZShapeInfo, void* extraArguments) {
  auto stream = lc->getCudaStream();
  auto sizeOf = sizeof(graph::RandomGenerator);
  Pointer stateDevice;

  cudaError_t res = cudaMalloc(reinterpret_cast<void**>(&stateDevice), sizeOf);
  checkCudaErrors(cudaStreamSynchronize(*stream));
  checkCudaErrors(cudaMemcpyAsync(stateDevice, stateHost, sizeOf, cudaMemcpyHostToDevice, *stream));

  auto rng = reinterpret_cast<graph::RandomGenerator*>(stateHost);

  dim3 launchDims = getLaunchDims("random");
  auto xType = ArrayOptions::dataType(hZShapeInfo);

  BUILD_SINGLE_SELECTOR(xType, functions::random::RandomFunction,
                        ::executeCudaTriple(launchDims, stream, opNum, stateDevice, dX, dXShapeInfo, dY, dYShapeInfo,
                                            dZ, dZShapeInfo, extraArguments),
                        SD_FLOAT_TYPES);

  res = cudaStreamSynchronize(*stream);
  if (res != 0) throw cuda_exception::build("execRandom XYZ failed", res);

  cudaFree(stateDevice);

  rng->rewindH(shape::length(hZShapeInfo));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3All(LaunchContext* lc, int opNum, const void* hX, const LongType* hXShapeInfo,
                                         const void* dX, const LongType* dXShapeInfo, void* extraParamsVals,
                                         const void* hY, const LongType* hYShapeInfo, const void* dY,
                                         const LongType* dYShapeInfo, void* hZ, const LongType* hZShapeInfo, void* dZ,
                                         const LongType* dZShapeInfo, LongType* dimension, LongType dimensionLength,
                                         const LongType* xTadShapeInfo, const LongType* xOffsets,
                                         const LongType* yTadShapeInfo, const LongType* yOffsets) {
  auto stream = lc->getCudaStream();
  auto allocationPointer = lc->getAllocationPointer();
  auto reductionPointer = lc->getReductionPointer();

  if (Environment::getInstance().isDebugAndVerbose()) printf("D119 opType:[%i]\n", opNum);

  dim3 launchDims = getReduceAllDims(shape::length(hZShapeInfo));

  if (Environment::getInstance().isVerbose() && launchDims.x == 1) printf("AD119 opType:[%i]\n", opNum);

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hYShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);

  if (yType != xType)
    throw datatype_exception::build("NativeOpExecutioner::execReduce3All both operands must have same data type",
                                        xType, yType);

  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::reduce3::Reduce3,
      ::execAll(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParamsVals, dZ, dZShapeInfo,
                dimension, dimensionLength, 1, allocationPointer, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);

  // TODO: remove after the release
  auto res = cudaStreamSynchronize(*stream);
  if (res != 0) throw cuda_exception::build("execReduce3All failed", res);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3TAD(LaunchContext* lc, int opNum, const void* hX, const LongType* hXShapeInfo,
                                         const void* dX, const LongType* dXShapeInfo, void* extraParamsVals,
                                         const void* hY, const LongType* hYShapeInfo, const void* dY,
                                         const LongType* dYShapeInfo, void* hZ, const LongType* hZShapeInfo, void* dZ,
                                         const LongType* dZShapeInfo, long long int* dimension,
                                         long long int dimensionLength, const LongType* tadShapeInfo,
                                         const LongType* tadOffsets, const LongType* yTadShapeInfo,
                                         const LongType* yTadOffsets) {
  if (shape::isScalar(hZShapeInfo)) {
    execReduce3(lc, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParamsVals, hY, hYShapeInfo, dY,
                                     dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo);
    return;
  }

  auto stream = lc->getCudaStream();
  auto allocationPointer = lc->getAllocationPointer();

  auto xType = ArrayOptions::dataType(hXShapeInfo);
  auto yType = ArrayOptions::dataType(hYShapeInfo);
  auto zType = ArrayOptions::dataType(hZShapeInfo);

  if (xType != yType)
    throw datatype_exception::build("NativeOpExecutioner::execReduce3TAD requires Y operand to have X type", xType,
                                        yType);

  if (!DataTypeUtils::isR(zType))
    throw datatype_exception::build(
        "NativeOpExecutioner::execReduce3TAD requires Z operand to have floating point data type", zType);

  auto numBlocks = shape::length(hZShapeInfo);
  dim3 launchDims = getReduceDims(numBlocks);

  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::reduce3::Reduce3,
      ::exec(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParamsVals, dZ, dZShapeInfo, dimension,
             dimensionLength, 1, allocationPointer, tadShapeInfo, tadOffsets, yTadShapeInfo, yTadOffsets),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);

  // TODO: remove after the release
  auto res = cudaStreamSynchronize(*stream);
  if (res != 0) throw cuda_exception::build("execReduce3TAD failed", res);
}
