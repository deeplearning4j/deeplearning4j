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
