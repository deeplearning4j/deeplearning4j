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
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceLongFunction,
                        ::execReduce(launchDims, stream, opNum, dX,
                                     const_cast<sd::LongType*>(dXShapeInfo),
                                     const_cast<sd::LongType*>(hXShapeInfo), extraParams,
                                     reductionPointer, dZ,
                                     const_cast<sd::LongType*>(dZShapeInfo),
                                     const_cast<sd::LongType*>(hZShapeInfo), dimension),
                        SD_COMMON_TYPES, SD_LONG_TYPES);
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
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceBoolFunction,
                        ::execReduce(launchDims, stream, opNum, dX, const_cast<sd::LongType*>(dXShapeInfo),
                                     const_cast<sd::LongType*>(hXShapeInfo), extraParams,
                                     reductionPointer, dZ,
                                     const_cast<sd::LongType*>(dZShapeInfo), const_cast<sd::LongType*>(hZShapeInfo), dimension),
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
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceFloatFunction,
                        ::execReduce(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams,
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
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::indexreduce::IndexReduce,
      ::executeIndexReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, shape::rank(hXShapeInfo), extraParams, dz,
                                 dZShapeInfo, 0, nullptr, 0, 1, allocationPointer, reductionPointer, nullptr, nullptr),
      SD_COMMON_TYPES, SD_INDEXING_TYPES);
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
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceFloatFunction,
                        ::execReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ,
                                           dZShapeInfo, hZShapeInfo, nullptr, 0, reductionPointer, nullptr),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
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
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceBoolFunction,
                        ::execReduceScalar(launchDims, stream, opNum, dX, dXShapeInfo, hXShapeInfo, extraParams, dZ,
                                           dZShapeInfo, hZShapeInfo, nullptr, 0, reductionPointer, nullptr),
                        SD_COMMON_TYPES, SD_BOOL_TYPES);
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
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce::ReduceLongFunction,
                        ::execReduceScalar(launchDims, stream, opNum, dX,
                                           const_cast<sd::LongType*>(dXShapeInfo),
                                           const_cast<sd::LongType*>(hXShapeInfo), extraParams, dZ,
                                           const_cast<sd::LongType*>(dZShapeInfo), const_cast<sd::LongType*>(hZShapeInfo), nullptr, 0, reductionPointer, nullptr),
                        SD_COMMON_TYPES, SD_LONG_TYPES);
}

////////////////////////////////////////////////////////////////////////
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
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3,
                        ::execScalar(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParams, dZ,
                                     dZShapeInfo, allocationPointer, reductionPointer, nullptr),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
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
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::reduce3::Reduce3,
      ::exec(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParamsVals, dZ, dZShapeInfo, dimension,
             dimensionLength, 1, allocationPointer, xTadOnlyShapeInfo, xTadOffsets, yTadOnlyShapeInfo, yTadOffsets),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);
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
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::reduce3::Reduce3,
                        ::execScalar(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParams, dZ,
                                     dZShapeInfo, allocationPointer, reductionPointer, nullptr),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
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
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::reduce3::Reduce3,
      ::execAll(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParamsVals, dZ, dZShapeInfo,
                dimension, dimensionLength, 1, allocationPointer, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);

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
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::reduce3::Reduce3,
      ::exec(launchDims, stream, opNum, dX, dXShapeInfo, dY, dYShapeInfo, extraParamsVals, dZ, dZShapeInfo, dimension,
             dimensionLength, 1, allocationPointer, tadShapeInfo, tadOffsets, yTadShapeInfo, yTadOffsets),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);

  auto res = cudaStreamSynchronize(*stream);
  if (res != 0) {
    std::string errorMessage = "execReduce3TAD failed with error code: " + std::to_string(static_cast<int>(res));
    THROW_EXCEPTION(errorMessage.c_str());
  }
}
