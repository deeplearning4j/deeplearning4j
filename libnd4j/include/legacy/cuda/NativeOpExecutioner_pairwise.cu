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
