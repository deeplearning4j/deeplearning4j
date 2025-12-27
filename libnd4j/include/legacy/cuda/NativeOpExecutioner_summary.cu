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
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::summarystats::SummaryStatsReduce,
      ::execSummaryStatsReduce(launchDims, stream, opNum, const_cast<void*>(dX), dXShapeInfo, hXShapeInfo, extraParams, dZ, dZShapeInfo,
                               hZShapeInfo, nullptr, nullptr, biasCorrected, reductionPointer),
      SD_COMMON_TYPES, SD_FLOAT_TYPES);
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

  // Now call the available signature without dimension parameters
  BUILD_DOUBLE_SELECTOR(xType, zType, functions::summarystats::SummaryStatsReduce,
                        ::execSummaryStatsReduce(launchDims, stream, opNum, const_cast<void*>(dX), dXShapeInfo, hXShapeInfo, extraParams,
                                                 dZ, dZShapeInfo, hZShapeInfo, computedTadShapeInfo,
                                                 computedTadOffsets, biasCorrected, reductionPointer),
                        SD_COMMON_TYPES, SD_FLOAT_TYPES);
}
//////////////////////////////////////////
