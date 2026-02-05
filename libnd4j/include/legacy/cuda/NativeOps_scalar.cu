/* ******************************************************************************
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
// Split from NativeOps.cu to reduce object file size for SD_GCC_FUNCTRACE builds
// Contains: execScalar, execScalarBool, execScalarTad, execScalarBoolTad
//

#include <cuda.h>
#include <exceptions/cuda_exception.h>
#include <execution/LaunchContext.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/DebugHelper.h>
#include <helpers/StringUtils.h>
#include <legacy/NativeOps.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/scalar.h>
#include <system/common.h>
#include <system/op_boilerplate.h>
#include <system/Environment.h>
#include <array/ArrayOptions.h>
#include <helpers/shape.h>
#include <execution/cuda/LaunchDims.h>

////////////////////////////////////////////////////////////////////////
void execScalarBool(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, void *extraParams) {
  try {
    x->prepareSpecialUse({z}, {x, scalar});

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execScalarBool(
        lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special(),
        shape::isEmptyConst(scalar->shapeInfo()) ? nullptr : scalar->buffer(),
        scalar->shapeInfo(),
        shape::isEmptyConst(scalar->shapeInfo()) ? nullptr : scalar->specialBuffer(),
        sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(scalar->shapeInfo())->special(), extraParams);

    x->registerSpecialUse({z}, {x, scalar});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execScalarBoolTad(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, void *extraParams, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x, scalar});
    dimension->preparePrimaryUse({}, {dimension});

    auto dim = dimension != nullptr ? reinterpret_cast<sd::LongType *>(dimension->buffer()) : nullptr;
    sd::LongType dimensionLength = static_cast<sd::LongType>(shape::length(dimension->shapeInfo()));

    auto xTadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), dim, dimensionLength);
    auto xTadShapeInfo = xTadPack->specialShapeInfo();
    auto xOffsets = xTadPack->specialOffsets();

    auto zTadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(), dim, dimensionLength);
    auto zTadShapeInfo = zTadPack->specialShapeInfo();
    auto zOffsets = zTadPack->specialOffsets();

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execScalarBool(
        lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special(),
        extraParams,
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special(),
        shape::isEmptyConst(scalar->shapeInfo()) ? nullptr : scalar->buffer(),
        scalar->shapeInfo(),
        shape::isEmptyConst(scalar->shapeInfo()) ? nullptr : scalar->specialBuffer(),
        sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(scalar->shapeInfo())->special(),
        dim, dimensionLength,
        xTadShapeInfo, xOffsets, zTadShapeInfo, zOffsets);

    x->registerSpecialUse({z}, {x, scalar});
    dimension->registerSpecialUse({}, {dimension});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
  }
}

////////////////////////////////////////////////////////////////////////
void execScalar(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, void *extraParams) {
  try {
    // Clear any stale CUDA errors from initialization before running operations
    cudaGetLastError();
    x->prepareSpecialUse({z}, {x, scalar});

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execScalar(
        lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special(),
        shape::isEmptyConst(scalar->shapeInfo()) ? nullptr : scalar->buffer(),
        scalar->shapeInfo(),
        shape::isEmptyConst(scalar->shapeInfo()) ? nullptr : scalar->specialBuffer(),
        sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(scalar->shapeInfo())->special(), extraParams);

    x->registerSpecialUse({z}, {x, scalar});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execScalarTad(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, void *extraParams, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x, scalar});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionPtr = dimension != nullptr ? reinterpret_cast<sd::LongType *>(dimension->buffer()) : nullptr;
    sd::LongType dimensionLength = static_cast<sd::LongType>(shape::length(dimension->shapeInfo()));

    auto xTadPack =
        sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), dimensionPtr, dimensionLength);
    auto xTadShapeInfo = xTadPack->specialShapeInfo();
    auto xOffsets = xTadPack->specialOffsets();

    auto zTadPack =
        sd::ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(), dimensionPtr, dimensionLength);
    auto zTadShapeInfo = zTadPack->specialShapeInfo();
    auto zOffsets = zTadPack->specialOffsets();

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);

    auto xType = sd::ArrayOptions::dataType(x->shapeInfo());
    auto yType = sd::ArrayOptions::dataType(scalar->shapeInfo());
    auto zType = sd::ArrayOptions::dataType(z->shapeInfo());

    dim3 launchDims = getLaunchDims("scalarTad");

    BUILD_SINGLE_SELECTOR_THRICE(
        xType, functions::scalar::ScalarTransform,
        ::executeCudaAlongDimension(
            launchDims, stream, opNum,
            shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
            xTadShapeInfo,
            shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
            zTadShapeInfo,
            shape::isEmptyConst(scalar->shapeInfo()) ? nullptr : scalar->specialBuffer(),
            extraParams, dimensionPtr, dimensionLength, xTadShapeInfo, xOffsets, zTadShapeInfo, zOffsets),
        SD_COMMON_TYPES);

    DEBUG_KERNEL(stream, opNum);

    x->registerSpecialUse({z}, {x, scalar});
    dimension->registerSpecialUse({}, {dimension});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
