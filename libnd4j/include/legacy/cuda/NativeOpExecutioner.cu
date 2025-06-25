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
void NativeOpExecutioner::execPairwiseTransform(sd::LaunchContext* lc, int opNum, void const* hX,
                                                sd::LongType const* hXShapeInfo, void const* dX,
                                                sd::LongType const* dXShapeInfo, void const* hY,
                                                sd::LongType const* hYShapeInfo, void const* dY,
                                                sd::LongType const* dYShapeInfo, void* hZ, sd::LongType const* hZShapeInfo,
                                                void* dZ, sd::LongType const* dZShapeInfo, void* extraParams) {
  auto stream = lc->getCudaStream();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    std::string errorMessage;
    errorMessage +=
        "NativeOpExecutioner::execPairwiseTransform:: unable to execute on strings. Please write logic "
        "higher level in each op for the string data type.";
    errorMessage += "X type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += "Y type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);
    errorMessage += "Z type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (xType != zType && yType != zType) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execPairwiseTransform both operands must have same data type";
    errorMessage += "X type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += "Y type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);
    errorMessage += "Z type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
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


  BUILD_SINGLE_SELECTOR_THRICE(
      xType, functions::pairwise_transforms::PairWiseTransform,
      ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraParams),
      SD_COMMON_TYPES)
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execPairwiseBoolTransform(sd::LaunchContext* lc, int opNum, void const* hX,
                                                    sd::LongType const* hXShapeInfo, void const* dX,
                                                    sd::LongType const* dXShapeInfo, void const* hY,
                                                    sd::LongType const* hYShapeInfo, void const* dY,
                                                    sd::LongType const* dYShapeInfo, void* hZ, sd::LongType const* hZShapeInfo,
                                                    void* dZ, sd::LongType const* dZShapeInfo, void* extraParams) {
  auto stream = lc->getCudaStream();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execPairwiseBoolTransform:: unable to execute on strings. Please write logic higher "
        "level in each op for the string data type.")
  }

  if (!sd::DataTypeUtils::isB(zType)) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execPairwiseBoolTransform requires Z operand to have BOOL type";
    errorMessage += "X type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += "Y type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);
    errorMessage += "Z type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (yType != xType) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execPairwiseBoolTransform both operands must have same data type";
    errorMessage += "X type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += "Y type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);

    THROW_EXCEPTION(errorMessage.c_str());
  }
  dim3 launchDims = getLaunchDims("pairwiseTransforms");
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::pairwise_transforms::PairWiseBoolTransform,
      ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraParams),
      SD_COMMON_TYPES, SD_BOOL_TYPES)
#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execPairwiseIntTransform(sd::LaunchContext* lc, int opNum, void const* hX,
                                                   sd::LongType const* hXShapeInfo, void const* dX,
                                                   sd::LongType const* dXShapeInfo, void const* hY,
                                                   sd::LongType const* hYShapeInfo, void const* dY,
                                                   sd::LongType const* dYShapeInfo, void* hZ, sd::LongType const* hZShapeInfo,
                                                   void* dZ, sd::LongType const* dZShapeInfo, void* extraParams) {
  auto stream = lc->getCudaStream();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    std::string errorMessage;
    errorMessage +=
        "NativeOpExecutioner::execPairwiseIntTransform:: unable to execute on strings. Please write logic "
        "higher level in each op for the string data type.";

    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (!sd::DataTypeUtils::isZ(zType)) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execPairwiseIntTransform requires Z operand to have INT type";
    errorMessage += "X type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += "Y type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);
    errorMessage += "Z type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (yType != xType || zType != xType) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execPairwiseIntTransform both operands must have same data type x type:";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += " y type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  dim3 launchDims = getLaunchDims("pairwiseTransforms");

  BUILD_SINGLE_SELECTOR(
      xType, functions::pairwise_transforms::PairWiseIntTransform,
      ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraParams),
      SD_INTEGER_TYPES)
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execSummaryStatsScalar(sd::LaunchContext* lc, int opNum, void const* hX,
                                                 sd::LongType* hXShapeInfo, void const* dX,
                                                 sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
                                                 sd::LongType* hZShapeInfo, void* dZ, sd::LongType* dZShapeInfo,
                                                 bool biasCorrected) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  dim3 launchDims = getLaunchDims("summaryStats");

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execSummaryStatsScalar:: unable to execute on strings. Please write logic higher level "
        "in each op for the string data type.")
  }
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::summarystats::SummaryStatsReduce,
      ::execSummaryStatsReduceScalar(launchDims, stream, opNum, const_cast<void*>(dX), dXShapeInfo, hXShapeInfo, extraParams, dZ,
                                     dZShapeInfo, hZShapeInfo, nullptr, nullptr, biasCorrected, reductionPointer),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);
#endif
}

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
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::broadcast::BroadcastBool,
      ::execBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraParams),
      SD_COMMON_TYPES, SD_BOOL_TYPES);
#endif
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
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::broadcast::BroadcastBool,
      ::execInverseBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, extraParams,
                             dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
      SD_COMMON_TYPES, SD_BOOL_TYPES)
#endif
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

#if SD_IS_TRIPLE_TYPE_COMPILED(xType,xType,xType)
  BUILD_SINGLE_SELECTOR_THRICE(
      xType, functions::broadcast::Broadcast,
      ::execBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, dimension,
                      dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
      SD_COMMON_TYPES);
#endif
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

#if SD_IS_TRIPLE_TYPE_COMPILED(xType,xType,xType)
  BUILD_SINGLE_SELECTOR_THRICE(
      xType, functions::broadcast::Broadcast,
      ::execBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo), SD_COMMON_TYPES);
#endif
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
#if SD_IS

  BUILD_SINGLE_SELECTOR_THRICE(
      xType, functions::broadcast::Broadcast,
      ::execInverseBroadcast(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, dZ, dZShapeInfo, dimension,
                             dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ),
      SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceSame(sd::LaunchContext* lc, int opNum, void const* hX, sd::LongType const* hXShapeInfo,
                                         void const* dX, sd::LongType const* dXShapeInfo, void* extraParams, void* hZ,
                                         sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo,
                                         sd::LongType* dimension, sd::LongType dimensionLength) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  if (sd::Environment::getInstance().isDebugAndVerbose()) printf("SF7 opType:[%i]\n", opNum);

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduceSame:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  if (zType != xType) {
    std::string errorMessage = "NativeOpExecutioner::execReduceSame requires both X & Z operands to have same type. X type: " + sd::DataTypeUtils::asString(xType) + ", Z type: " + sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  auto numBlocks = shape::length(hZShapeInfo);
  dim3 launchDims = getReduceDims(numBlocks);

  BUILD_SINGLE_SELECTOR(xType, functions::reduce::ReduceSameFunction,
                        ::execReduce(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams,
                                     reductionPointer, dZ, dZShapeInfo, hZShapeInfo, dimension),
                        SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceLong(sd::LaunchContext* lc, int opNum, void const* hX, sd::LongType const* hXShapeInfo,
                                         void const* dX, sd::LongType const* dXShapeInfo, void* extraParams, void* hZ,
                                         sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo,
                                         sd::LongType* dimension, sd::LongType dimensionLength) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();
  sd::LongType* allocationPointer = lc->getAllocationPointer();

  if (sd::Environment::getInstance().isDebugAndVerbose()) printf("LF7 opType:[%i]\n", opNum);

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduceLong:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  if (zType != sd::INT64) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execReduceLong requires Z operand to have INT64 type";
    errorMessage += "X type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += "Y type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    errorMessage += "Z type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  auto numBlocks = shape::length(hZShapeInfo);
  dim3 launchDims = getReduceDims(numBlocks);
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceLongFunction,
                        ::execReduce(launchDims, stream, opNum, dX,
                                     const_cast<sd::LongType*>(dXShapeInfo),
                                     const_cast<sd::LongType*>(hXShapeInfo), extraParams,
                                     reductionPointer, dZ,
                                     const_cast<sd::LongType*>(dZShapeInfo),
                                     const_cast<sd::LongType*>(hZShapeInfo), dimension),
                        SD_COMMON_TYPES, SD_LONG_TYPES);
#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceBool(sd::LaunchContext* lc, int opNum, void const* hX, sd::LongType const* hXShapeInfo,
                                         void const* dX, sd::LongType const* dXShapeInfo, void* extraParams, void* hZ,
                                         sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo,
                                         sd::LongType* dimension, sd::LongType dimensionLength) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  if (sd::Environment::getInstance().isDebugAndVerbose()) printf("BF7 opType:[%i]\n", opNum);

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduceBool:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  if (zType != sd::BOOL)
    THROW_EXCEPTION("NativeOpExecutioner::execReduceBool requires Z operand to have BOOL type");

  auto numBlocks = shape::length(hZShapeInfo);
  dim3 launchDims = getReduceDims(numBlocks);
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceBoolFunction,
                        ::execReduce(launchDims, stream, opNum, dX, const_cast<sd::LongType*>(dXShapeInfo),
                                     const_cast<sd::LongType*>(hXShapeInfo), extraParams,
                                     reductionPointer, dZ,
                                     const_cast<sd::LongType*>(dZShapeInfo), const_cast<sd::LongType*>(hZShapeInfo), dimension),
                        SD_COMMON_TYPES, SD_BOOL_TYPES);
#endif
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
void NativeOpExecutioner::execReduceFloat(sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXShapeInfo,
                                          const void* dX, const sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
                                          const sd::LongType* hZShapeInfo, void* dZ, const sd::LongType* dZShapeInfo,
                                          sd::LongType* dimension, sd::LongType dimensionLength) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();


  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduceFloat:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  auto numBlocks = shape::length(hZShapeInfo);
  dim3 launchDims = getReduceDims(numBlocks);
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceFloatFunction,
                        ::execReduce(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams,
                                     reductionPointer, dZ, dZShapeInfo, hZShapeInfo, dimension),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
#endif
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
void NativeOpExecutioner::execIndexReduce(sd::LaunchContext* lc, int opNum, void const* hX, sd::LongType const* hXShapeInfo,
                                          void const* dX, sd::LongType const* dXShapeInfo, void* extraParams, void* hZ,
                                          sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo,
                                          sd::LongType* dimension, sd::LongType dimensionLength, sd::LongType const* tadShapeInfo,
                                          sd::LongType const* tadOffsets) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();
  auto allocationPointer = lc->getAllocationPointer();

  if (sd::Environment::getInstance().isDebugAndVerbose()) {
    printf("F2 opType:[%i]\n", opNum);
  }
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execIndexReduce:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  auto numBlocks = shape::length(hZShapeInfo);
  auto tadLength = shape::length(hXShapeInfo) / numBlocks;
  dim3 launchDims = getReduceDims(numBlocks);
  if (zType != sd::INT64 && zType != sd::INT32) {
    std::string errorMessage = "NativeOpExecutioner::execIndexReduce requires Z operand to have INT32/INT64 type. Z type: " + sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  auto dz = reinterpret_cast<sd::LongType*>(dZ);
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
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
#endif
}

/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param extraParams
 */
////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execIndexReduceScalar(sd::LaunchContext* lc, int opNum, void const* hX,
                                                sd::LongType const* hXShapeInfo, void const* dX,
                                                sd::LongType const* dXShapeInfo, void* extraParams, void* hZ,
                                                sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();
  sd::LongType* allocationPointer = lc->getAllocationPointer();

  auto xLength = shape::length(hXShapeInfo);
  dim3 launchDims = getReduceDims(xLength);

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execIndexReduceScalar:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }

  if (zType != sd::INT64 && zType != sd::INT32) {
    std::string errorMessage = "NativeOpExecutioner::execIndexReduceScalar requires Z operand to have INT32/INT64 data type. Z type: " + sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  auto dz = reinterpret_cast<sd::LongType*>(dZ);
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::indexreduce::IndexReduce,
      ::executeIndexReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, shape::rank(hXShapeInfo), extraParams, dz,
                                 dZShapeInfo, 0, nullptr, 0, 1, allocationPointer, reductionPointer, nullptr, nullptr),
      SD_COMMON_TYPES, SD_INDEXING_TYPES);
#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceFloatScalar(sd::LaunchContext* lc, int opNum, void const* hX,
                                                sd::LongType const* hXShapeInfo, void const* dX,
                                                sd::LongType const* dXShapeInfo, void* extraParams, void* hZ,
                                                sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduceFloatScalar:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }
  auto xLength = shape::length(hXShapeInfo);
  dim3 launchDims = getReduceDims(xLength);
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceFloatFunction,
                        ::execReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ,
                                           dZShapeInfo, hZShapeInfo, nullptr, 0, reductionPointer, nullptr),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceBoolScalar(sd::LaunchContext* lc, int opNum, void const* hX,
                                               sd::LongType const* hXShapeInfo, void const* dX, sd::LongType const* dXShapeInfo,
                                               void* extraParams, void* hZ, sd::LongType const* hZShapeInfo, void* dZ,
                                               sd::LongType const* dZShapeInfo) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduceBoolScalar:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }
  if (zType != sd::BOOL)
    THROW_EXCEPTION("NativeOpExecutioner::execReduceBoolScalar requires Z operand to have BOOL type");

  auto xLength = shape::length(hXShapeInfo);
  dim3 launchDims = getReduceDims(xLength);
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceBoolFunction,
                        ::execReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ,
                                           dZShapeInfo, hZShapeInfo, nullptr, 0, reductionPointer, nullptr),
                        SD_COMMON_TYPES, SD_BOOL_TYPES);
#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceSameScalar(sd::LaunchContext* lc, int opNum, void const* hX,
                                               sd::LongType const* hXShapeInfo, void const* dX, sd::LongType const* dXShapeInfo,
                                               void* extraParams, void* hZ, sd::LongType const* hZShapeInfo, void* dZ,
                                               sd::LongType const* dZShapeInfo) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduceSameScalar:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }
  if (zType != xType) {
    std::string errorMessage = "NativeOpExecutioner::execReduceSameScalar requires both X & Z operands to have same type. X type: " + sd::DataTypeUtils::asString(xType) + ", Z type: " + sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  auto xLength = shape::length(hXShapeInfo);
  dim3 launchDims = getReduceDims(xLength);

  BUILD_SINGLE_SELECTOR(xType, functions::reduce::ReduceSameFunction,
                        ::execReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ,
                                           dZShapeInfo, hZShapeInfo, nullptr, 0, reductionPointer, nullptr),
                        SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduceLongScalar(sd::LaunchContext* lc, int opNum, void const* hX,
                                               sd::LongType const* hXShapeInfo, void const* dX, sd::LongType const* dXShapeInfo,
                                               void* extraParams, void* hZ, sd::LongType const* hZShapeInfo, void* dZ,
                                               sd::LongType const* dZShapeInfo) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduceLongScalar:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }
  if (zType != sd::INT64) {
    std::string errorMessage = "NativeOpExecutioner::execReduceLongScalar wrong Z data type. Expected: " + sd::DataTypeUtils::asString(sd::INT64) + ", but got: " + sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  auto xLength = shape::length(hXShapeInfo);
  dim3 launchDims = getReduceDims(xLength);
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceLongFunction,
                        ::execReduceScalar(launchDims, stream, opNum, dX,
                                           const_cast<sd::LongType*>(dXShapeInfo),
                                           const_cast<sd::LongType*>(hXShapeInfo), extraParams, dZ,
                                           const_cast<sd::LongType*>(dZShapeInfo), const_cast<sd::LongType*>(hZShapeInfo), nullptr, 0, reductionPointer, nullptr),
                        SD_COMMON_TYPES, SD_LONG_TYPES);
#endif
}

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
#if SD_IS_PAIR_TYPE_COMPILED(xType,xType)
  BUILD_DOUBLE_SELECTOR(xType,xType, functions::transform::TransformFloat,
                        ::executeTransformShaped(launchDims, stream,  opNum, dX,
                                                 dXShapeInfo, xRank,extraParams, dZ,
                                                 dZShapeInfo, zRank,
                                                 nullptr,nullptr,
                                                 nullptr,nullptr),
                        SD_COMMON_TYPES,SD_COMMON_TYPES);
#endif
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
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
    BUILD_DOUBLE_SELECTOR(xType, zType, functions::transform::TransformAny,
                          ::executeTransformShaped(launchDims, stream, opNum, dX, dXShapeInfo, xRank, extraParams, dZ,
                                                   dZShapeInfo, zRank, nullptr, nullptr, nullptr, nullptr),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);
#endif
  }
}



////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execSummaryStats(sd::LaunchContext* lc, int opNum, void const* hX,
                                           sd::LongType* hXShapeInfo, void const* dX, sd::LongType* dXShapeInfo,
                                           void* extraParams, void* hZ, sd::LongType* hZShapeInfo, void* dZ,
                                           sd::LongType* dZShapeInfo, bool biasCorrected) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  dim3 launchDims = getLaunchDims("summaryStats");

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execSummaryStats:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  if (!sd::DataTypeUtils::isR(zType)) {
    std::string errorMessage = "NativeOpExecutioner::execSummaryStats requires Z operand to have floating point data type. Z type: " + sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::summarystats::SummaryStatsReduce,
      ::execSummaryStatsReduce(launchDims, stream, opNum, const_cast<void*>(dX), dXShapeInfo, hXShapeInfo, extraParams, dZ, dZShapeInfo,
                               hZShapeInfo, nullptr, nullptr, biasCorrected, reductionPointer),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);
#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execSummaryStats(sd::LaunchContext* lc, int opNum, void const* hX,
                                           sd::LongType* hXShapeInfo, void const* dX, sd::LongType* dXShapeInfo,
                                           void* extraParams, void* hZ, sd::LongType* hZShapeInfo, void* dZ,
                                           sd::LongType* dZShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength,
                                           sd::LongType* tadShapeInfo, sd::LongType* tadOffsets,
                                           bool biasCorrected) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();

  dim3 launchDims = getLaunchDims("summaryStats");

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execSummaryStats:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  if (!sd::DataTypeUtils::isR(zType)) {
    std::string errorMessage = "NativeOpExecutioner::execSummaryStats requires Z operand to have floating point data type. Z type: " + sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  // First, compute TAD shape info if needed based on dimensions
  sd::LongType* computedTadShapeInfo = tadShapeInfo;
  sd::LongType* computedTadOffsets = tadOffsets;

  // If tadShapeInfo is not provided but dimensions are, compute them
  if (dimensionLength > 0 && dimension != nullptr && tadShapeInfo == nullptr) {
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);
    computedTadShapeInfo = tadPack->specialShapeInfo();
    computedTadOffsets = tadPack->specialOffsets();
  }

#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  // Now call the available signature without dimension parameters
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::summarystats::SummaryStatsReduce,
                        ::execSummaryStatsReduce(launchDims, stream, opNum, const_cast<void*>(dX), dXShapeInfo, hXShapeInfo, extraParams,
                                                 dZ, dZShapeInfo, hZShapeInfo, computedTadShapeInfo,
                                                 computedTadOffsets, biasCorrected, reductionPointer),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
#endif
}
//////////////////////////////////////////
void NativeOpExecutioner::execReduce3(sd::LaunchContext* lc, int opNum, void const* hX, sd::LongType const* hXShapeInfo,
                                      void const* dX, sd::LongType const* dXShapeInfo, void* extraParams, void const* hY,
                                      sd::LongType const* hYShapeInfo, void const* dY, sd::LongType const* dYShapeInfo,
                                      void* hZ, sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo) {
  auto stream = lc->getCudaStream();
  auto reductionPointer = lc->getReductionPointer();
  auto allocationPointer = lc->getAllocationPointer();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduce3:: unable to execute on strings. Please write logic higher level in each op "
        "for the string data type.")
  }

  dim3 launchDims = getReduceDims(shape::length(hXShapeInfo));

  if (xType != yType) {
    std::string errorMessage = "NativeOpExecutioner::execReduce3 requires Y operand to have X type. X type: " + sd::DataTypeUtils::asString(xType) + ", Y type: " + sd::DataTypeUtils::asString(yType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (!sd::DataTypeUtils::isR(zType)) {
    std::string errorMessage = "NativeOpExecutioner::execReduce3 requires Z operand to have floating point data type. Z type: " + sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3,
                        ::execScalar(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParams, dZ,
                                     dZShapeInfo, allocationPointer, reductionPointer, nullptr),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3(sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXShapeInfo,
                                      const void* dX, const sd::LongType* dXShapeInfo, void* extraParamsVals,
                                      const void* hY, const sd::LongType* hYShapeInfo, const void* dY,
                                      const sd::LongType* dYShapeInfo, void* hZ, const sd::LongType* hZShapeInfo, void* dZ,
                                      const sd::LongType* dZShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength,
                                      const sd::LongType* xTadOnlyShapeInfo, const sd::LongType* xTadOffsets,
                                      const sd::LongType* yTadOnlyShapeInfo, const sd::LongType* yTadOffsets) {
  if (shape::isScalar(hZShapeInfo)) {
    execReduce3(lc, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParamsVals, hY, hYShapeInfo, dY,
                dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo);
    return;
  }

  auto stream = lc->getCudaStream();
  auto allocationPointer = lc->getAllocationPointer();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduce3:: unable to execute on strings. Please write logic higher level in each op "
        "for the string data type.")
  }
  if (xType != yType) {
    std::string errorMessage = "NativeOpExecutioner::execReduce3 requires Y operand to have X type. X type: " + sd::DataTypeUtils::asString(xType) + ", Y type: " + sd::DataTypeUtils::asString(yType);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (!sd::DataTypeUtils::isR(zType)) {
    std::string errorMessage = "NativeOpExecutioner::execReduce3 requires Z operand to have floating point data type. Z type: " + sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  auto numBlocks = shape::length(hZShapeInfo);
  dim3 launchDims = getReduceDims(numBlocks);
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::reduce3::Reduce3,
      ::exec(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParamsVals, dZ, dZShapeInfo, dimension,
             dimensionLength, 1, allocationPointer, xTadOnlyShapeInfo, xTadOffsets, yTadOnlyShapeInfo, yTadOffsets),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);
#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3Scalar(sd::LaunchContext* lc, int opNum, void const* hX,
                                            sd::LongType const* hXShapeInfo, void const* dX, sd::LongType const* dXShapeInfo,
                                            void* extraParams, void const* hY, sd::LongType const* hYShapeInfo,
                                            void const* dY, sd::LongType const* dYShapeInfo, void* hZ,
                                            sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo) {
  auto stream = lc->getCudaStream();
  auto allocationPointer = lc->getAllocationPointer();
  auto reductionPointer = lc->getReductionPointer();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execReduce3Scalar:: unable to execute on strings. Please write logic higher level in "
        "each op for the string data type.")
  }

  dim3 launchDims = getReduceDims(shape::length(hXShapeInfo));

  if (xType != yType) {
    std::string errorMessage = "NativeOpExecutioner::execReduce3Scalar requires Y operand to have X type. X type: " + sd::DataTypeUtils::asString(xType) + ", Y type: " + sd::DataTypeUtils::asString(yType);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (!sd::DataTypeUtils::isR(zType)) {
    std::string errorMessage = "NativeOpExecutioner::execReduce3Scalar requires Z operand to have floating point data type. Z type: " + sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3,
                        ::execScalar(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParams, dZ,
                                     dZShapeInfo, allocationPointer, reductionPointer, nullptr),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarBool(sd::LaunchContext* lc, int opNum, void const* hX, sd::LongType const* hXShapeInfo,
                                         void const* dX, sd::LongType const* dXShapeInfo, void* hZ,
                                         sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo,
                                         void const* hScalar, sd::LongType const* hScalarShapeInfo, void const* dScalar,
                                         sd::LongType const* dScalarShapeInfo, void* extraParams, bool allowParallelism) {
  auto stream = lc->getCudaStream();

  dim3 launchDims = getLaunchDims("scalarScan");

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execScalarBool:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }
  if (xType != yType) THROW_EXCEPTION("NativeOpExecutioner::execScalarBool requires X & Y to have same type");

  if (!sd::DataTypeUtils::isB(zType))
    THROW_EXCEPTION("NativeOpExecutioner::execScalarBool requires Z operand to have BOOL type");
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::scalar::ScalarBoolTransform,
      ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalar, extraParams),
      SD_COMMON_TYPES, SD_BOOL_TYPES);
#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarBool(sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXShapeInfo,
                                         const void* dX, const sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
                                         const sd::LongType* hZShapeInfo, void* dZ, const sd::LongType* dZShapeInfo,
                                         const void* hScalars, const sd::LongType* hScalarShapeInfo, const void* dScalars,
                                         const sd::LongType* dScalarShapeInfo, sd::LongType* dimension,
                                         sd::LongType dimensionLength, const sd::LongType* tadShapeInfo,
                                         const sd::LongType* tadOffsets, const sd::LongType* tadShapeInfoZ,
                                         const sd::LongType* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  dim3 launchDims = getLaunchDims("scalarScan");

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execScalarBool:: unable to execute on strings. Please write logic higher level in each "
        "op for the string data type.")
  }

  if (xType != yType) THROW_EXCEPTION("NativeOpExecutioner::execScalarBool requires X & Y to have same type");

  if (!sd::DataTypeUtils::isB(zType))
    THROW_EXCEPTION("NativeOpExecutioner::execScalarBool requires Z operand to have BOOL type");
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::scalar::ScalarBoolTransform,
      ::executeCudaAlongDimension(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalars, extraParams,
                                  dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ),
      SD_COMMON_TYPES, SD_BOOL_TYPES);
#endif
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarInt(sd::LaunchContext* lc, int opNum, void const* hX, sd::LongType const* hXShapeInfo,
                                        void const* dX, sd::LongType const* dXShapeInfo, void* hZ,
                                        sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo,
                                        void const* hScalar, sd::LongType const* hScalarShapeInfo, void const* dScalar,
                                        sd::LongType const* dScalarShapeInfo, void* extraParams, bool allowParallelism) {
  auto stream = lc->getCudaStream();

  dim3 launchDims = getLaunchDims("scalarScan");

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execScalarInt:: unable to execute on strings. Please write logic higher level in each op "
        "for the string data type.")
  }

  if (xType != yType || zType != xType)
    THROW_EXCEPTION("NativeOpExecutioner::execScalarInt requires X & Y to have same type");

  if (!sd::DataTypeUtils::isZ(zType))
    THROW_EXCEPTION("NativeOpExecutioner::execScalarInt requires Z operand to have INT type");

  BUILD_SINGLE_SELECTOR(
      xType, functions::scalar::ScalarIntTransform,
      ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalar, extraParams),
      SD_INTEGER_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalarInt(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                        const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                        const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                        const void *hScalars, const sd::LongType *hScalarShapeInfo, const void *dScalars,
                                        const sd::LongType *dScalarShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength,
                                        const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets,
                                        const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  dim3 launchDims = getLaunchDims("scalarScan");

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
    THROW_EXCEPTION(
        "NativeOpExecutioner::execScalarInt:: unable to execute on strings. Please write logic higher level in each op "
        "for the string data type.")
  }
  if (xType != yType || zType != xType)
    THROW_EXCEPTION("NativeOpExecutioner::execScalarInt requires X & Y to have same type");

  if (!sd::DataTypeUtils::isZ(zType))
    THROW_EXCEPTION("NativeOpExecutioner::execScalarInt requires Z operand to have INT type");

  BUILD_SINGLE_SELECTOR(
      xType, functions::scalar::ScalarIntTransform,
      ::executeCudaAlongDimension(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalars, extraParams,
                                  dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ),
      SD_INTEGER_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execScalar(sd::LaunchContext* lc, int opNum, void const* hX, sd::LongType const* hXShapeInfo,
                                     void const* dX, sd::LongType const* dXShapeInfo, void* hZ, sd::LongType const* hZShapeInfo,
                                     void* dZ, sd::LongType const* dZShapeInfo, void const* hScalar,
                                     sd::LongType const* hScalarShapeInfo, void const* dScalar,
                                     sd::LongType const* dScalarShapeInfo, void* extraParams, bool allowParallelism) {
  auto stream = lc->getCudaStream();

  dim3 launchDims = getLaunchDims("scalarScan");

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  if (sd::DataTypeUtils::isS(xType) || sd::DataTypeUtils::isS(yType) || sd::DataTypeUtils::isS(zType)) {
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
void NativeOpExecutioner::execScalar(sd::LaunchContext* lc, int opNum, void const* hX, sd::LongType const* hXShapeInfo,
                                     void const* dX, sd::LongType const* dXShapeInfo, void* extraParams, void* hZ,
                                     sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo,
                                     void const* hScalars, sd::LongType const* hScalarShapeInfo, void const* dScalars,
                                     sd::LongType const* dScalarShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength,
                                     sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets,
                                     sd::LongType const* tadShapeInfoZ, sd::LongType const* tadOffsetsZ) {
  auto stream = lc->getCudaStream();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hScalarShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  dim3 launchDims = getLaunchDims("scalarScan");


  if (xType != yType || xType != zType) {
    std::string errorMessage;
    errorMessage += "NativeOpExecutioner::execScalar requires both X & Y to have same data type";
    errorMessage += "X data type: ";
    errorMessage += sd::DataTypeUtils::asString(xType);
    errorMessage += ", Y data type: ";
    errorMessage += sd::DataTypeUtils::asString(yType);
    errorMessage += ", Z data type: ";
    errorMessage += sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (sd::DataTypeUtils::isS(xType)) {
#if defined(HAS_UTF8) || defined(HAS_UTF16) || defined(HAS_UTF32)
     BUILD_SINGLE_SELECTOR_THRICE(
        xType, functions::scalar::ScalarTransform,
        ::executeCudaAlongDimension(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalars, extraParams,
                                    dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ),
        SD_STRING_TYPES);
#endif
  } else {
    BUILD_SINGLE_SELECTOR_THRICE(
        xType, functions::scalar::ScalarTransform,
        ::executeCudaAlongDimension(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalars, extraParams,
                                    dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ),
        SD_COMMON_TYPES);
  }


  // TODO: remove after the release
  auto res = cudaStreamSynchronize(*stream);
  if (res != 0) {
    std::string errorMessage = "execScalar B failed with error code: " + std::to_string(static_cast<int>(res));
    THROW_EXCEPTION(errorMessage.c_str());
  }}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(sd::LaunchContext* lc, int opNum, sd::Pointer stateHost, void* hZ,
                                     sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo,
                                     void* extraArguments) {
  auto stream = lc->getCudaStream();
  auto sizeOf = sizeof(sd::graph::RandomGenerator);
  sd::Pointer stateDevice;

  cudaError_t res = cudaMalloc(reinterpret_cast<void**>(&stateDevice), sizeOf);
  checkCudaErrors(cudaStreamSynchronize(*stream));
  checkCudaErrors(cudaMemcpyAsync(stateDevice, stateHost, sizeOf, cudaMemcpyHostToDevice, *stream));

  dim3 launchDims = getLaunchDims("random");
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  auto rng = reinterpret_cast<sd::graph::RandomGenerator*>(stateHost);

  BUILD_SINGLE_SELECTOR(zType, functions::random::RandomFunction,
                        ::executeCudaSingle(launchDims, stream, opNum, stateDevice, dZ, dZShapeInfo, extraArguments),
                        SD_FLOAT_TYPES);

  res = cudaStreamSynchronize(*stream);
  if (res != 0) {
    std::string errorMessage = "execRandom X failed with error code: " + std::to_string(static_cast<int>(res));
    THROW_EXCEPTION(errorMessage.c_str());
  }
  cudaFree(stateDevice);

  rng->rewindH(shape::length(hZShapeInfo));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(sd::LaunchContext* lc, int opNum, sd::Pointer stateHost, void const* hX,
                                     sd::LongType const* hXShapeInfo, void const* dX, sd::LongType const* dXShapeInfo, void* hZ,
                                     sd::LongType const* hZShapeInfo, void* dZ, sd::LongType const* dZShapeInfo,
                                     void* extraArguments) {
  auto stream = lc->getCudaStream();

  auto sizeOf = sizeof(sd::graph::RandomGenerator);
  sd::Pointer stateDevice;

  cudaError_t res = cudaMalloc(reinterpret_cast<void**>(&stateDevice), sizeOf);
  checkCudaErrors(cudaStreamSynchronize(*stream));
  checkCudaErrors(cudaMemcpyAsync(stateDevice, stateHost, sizeOf, cudaMemcpyHostToDevice, *stream));

  auto rng = reinterpret_cast<sd::graph::RandomGenerator*>(stateHost);

  dim3 launchDims = getLaunchDims("random");
  auto xType = sd::ArrayOptions::dataType(hZShapeInfo);

  BUILD_SINGLE_SELECTOR(
      xType, functions::random::RandomFunction,
      ::executeCudaDouble(launchDims, stream, opNum, stateDevice, dX, dXShapeInfo, dZ, dZShapeInfo, extraArguments),
      SD_FLOAT_TYPES);

  res = cudaStreamSynchronize(*stream);
  if (res != 0) {
    std::string errorMessage = "execRandom XY failed with error code: " + std::to_string(static_cast<int>(res));
    THROW_EXCEPTION(errorMessage.c_str());
  }
  cudaFree(stateDevice);

  rng->rewindH(shape::length(hZShapeInfo));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execRandom(sd::LaunchContext* lc, int opNum, sd::Pointer stateHost, void const* hX,
                                     sd::LongType const* hXShapeInfo, void const* dX, sd::LongType const* dXShapeInfo,
                                     void const* hY, sd::LongType const* hYShapeInfo, void const* dY,
                                     sd::LongType const* dYShapeInfo, void* hZ, sd::LongType const* hZShapeInfo, void* dZ,
                                     sd::LongType const* dZShapeInfo, void* extraArguments) {
  auto stream = lc->getCudaStream();
  auto sizeOf = sizeof(sd::graph::RandomGenerator);
  sd::Pointer stateDevice;

  cudaError_t res = cudaMalloc(reinterpret_cast<void**>(&stateDevice), sizeOf);
  checkCudaErrors(cudaStreamSynchronize(*stream));
  checkCudaErrors(cudaMemcpyAsync(stateDevice, stateHost, sizeOf, cudaMemcpyHostToDevice, *stream));

  auto rng = reinterpret_cast<sd::graph::RandomGenerator*>(stateHost);

  dim3 launchDims = getLaunchDims("random");
  auto xType = sd::ArrayOptions::dataType(hZShapeInfo);

  BUILD_SINGLE_SELECTOR(xType, functions::random::RandomFunction,
                        ::executeCudaTriple(launchDims, stream, opNum, stateDevice, dX, dXShapeInfo, dY, dYShapeInfo,
                                            dZ, dZShapeInfo, extraArguments),
                        SD_FLOAT_TYPES);

  res = cudaStreamSynchronize(*stream);
  if (res != 0) {
    std::string errorMessage = "execRandom XYZ failed with error code: " + std::to_string(static_cast<int>(res));
    THROW_EXCEPTION(errorMessage.c_str());
  }
  cudaFree(stateDevice);

  rng->rewindH(shape::length(hZShapeInfo));
}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3All(sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXShapeInfo,
                                         const void* dX, const sd::LongType* dXShapeInfo, void* extraParamsVals,
                                         const void* hY, const sd::LongType* hYShapeInfo, const void* dY,
                                         const sd::LongType* dYShapeInfo, void* hZ, const sd::LongType* hZShapeInfo, void* dZ,
                                         const sd::LongType* dZShapeInfo, sd::LongType* dimension, sd::LongType dimensionLength,
                                         const sd::LongType* xTadShapeInfo, const sd::LongType* xOffsets,
                                         const sd::LongType* yTadShapeInfo, const sd::LongType* yOffsets) {
  auto stream = lc->getCudaStream();
  auto allocationPointer = lc->getAllocationPointer();
  auto reductionPointer = lc->getReductionPointer();

  if (sd::Environment::getInstance().isDebugAndVerbose()) printf("D119 opType:[%i]\n", opNum);

  dim3 launchDims = getReduceAllDims(shape::length(hZShapeInfo));

  if (sd::Environment::getInstance().isVerbose() && launchDims.x == 1) printf("AD119 opType:[%i]\n", opNum);

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  if (yType != xType) {
    std::string errorMessage = "NativeOpExecutioner::execReduce3All both operands must have same data type. X data type: "
                               + sd::DataTypeUtils::asString(xType) + ", Y data type: " + sd::DataTypeUtils::asString(yType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::reduce3::Reduce3,
      ::execAll(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParamsVals, dZ, dZShapeInfo,
                dimension, dimensionLength, 1, allocationPointer, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);
#endif

  auto res = cudaStreamSynchronize(*stream);
  if (res != 0) {
    std::string errorMessage = "execReduce3All failed with error code: " + std::to_string(static_cast<int>(res));
    THROW_EXCEPTION(errorMessage.c_str());
  }}

////////////////////////////////////////////////////////////////////////
void NativeOpExecutioner::execReduce3TAD(sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXShapeInfo,
                                         const void* dX, const sd::LongType* dXShapeInfo, void* extraParamsVals,
                                         const void* hY, const sd::LongType* hYShapeInfo, const void* dY,
                                         const sd::LongType* dYShapeInfo, void* hZ, const sd::LongType* hZShapeInfo, void* dZ,
                                         const sd::LongType* dZShapeInfo, sd::LongType* dimension,
                                         long long int dimensionLength, const sd::LongType* tadShapeInfo,
                                         const sd::LongType* tadOffsets, const sd::LongType* yTadShapeInfo,
                                         const sd::LongType* yTadOffsets) {
  if (shape::isScalar(hZShapeInfo)) {
    execReduce3(lc, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParamsVals, hY, hYShapeInfo, dY,
                dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo);
    return;
  }

  auto stream = lc->getCudaStream();
  auto allocationPointer = lc->getAllocationPointer();

  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);
  auto yType = sd::ArrayOptions::dataType(hYShapeInfo);
  auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

  if (xType != yType) {
    std::string errorMessage = "NativeOpExecutioner::execReduce3TAD requires Y operand to have X type. X data type: "
                               + sd::DataTypeUtils::asString(xType) + ", Y data type: " + sd::DataTypeUtils::asString(yType);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (!sd::DataTypeUtils::isR(zType)) {
    std::string errorMessage = "NativeOpExecutioner::execReduce3TAD requires Z operand to have floating point data type. Z data type: "
                               + sd::DataTypeUtils::asString(zType);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  auto numBlocks = shape::length(hZShapeInfo);
  dim3 launchDims = getReduceDims(numBlocks);
#if SD_IS_PAIR_TYPE_COMPILED(xType,zType)
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::reduce3::Reduce3,
      ::exec(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParamsVals, dZ, dZShapeInfo, dimension,
             dimensionLength, 1, allocationPointer, tadShapeInfo, tadOffsets, yTadShapeInfo, yTadOffsets),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);
#endif

  auto res = cudaStreamSynchronize(*stream);
  if (res != 0) {
    std::string errorMessage = "execReduce3TAD failed with error code: " + std::to_string(static_cast<int>(res));
    THROW_EXCEPTION(errorMessage.c_str());
  }
}
