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
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::scalar::ScalarBoolTransform,
      ::executeCudaShaped(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalar, extraParams),
      SD_COMMON_TYPES, SD_BOOL_TYPES);
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
  BUILD_DOUBLE_SELECTOR(
      xType, zType, functions::scalar::ScalarBoolTransform,
      ::executeCudaAlongDimension(launchDims, stream, opNum, dX, dXShapeInfo, dZ, dZShapeInfo, dScalars, extraParams,
                                  dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ),
      SD_COMMON_TYPES, SD_BOOL_TYPES);
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
