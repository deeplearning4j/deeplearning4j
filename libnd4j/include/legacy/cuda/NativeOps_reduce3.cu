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
// Contains: execReduce3, execReduce3Tad, execReduce3Scalar, execReduce3All
//

#include <cuda.h>

#include <execution/LaunchContext.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/DebugHelper.h>
#include <legacy/NativeOps.h>
#include <legacy/NativeOpExecutioner.h>
#include <system/common.h>
#include <array/ArrayOptions.h>
#include <helpers/shape.h>

////////////////////////////////////////////////////////////////////////
void execReduce3(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray y, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x, y});

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduce3(
        lc,
        opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(),
        extraParams,
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->buffer(),
        y->shapeInfo(),
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->specialBuffer(),
        y->specialShapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo());

    x->registerSpecialUse({z}, {x, y});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduce3Tad(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray y, OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x, y});
    dimension->preparePrimaryUse({}, {dimension});

    auto dim = dimension != nullptr ? reinterpret_cast<sd::LongType *>(dimension->buffer()) : nullptr;
    sd::LongType dimensionLength = static_cast<sd::LongType>(shape::length(dimension->shapeInfo()));

    // TAD-decompose x by the specified dimensions.
    // For y, only create a TAD pack if all requested dimensions are within y's rank.
    // When y has a lower rank than x (e.g. rank-1 needle vs rank-2 x), the dimensions
    // used to TAD x are out-of-bounds for y — pass nullptr for yTad pointers instead.
    // The CUDA transform kernel handles nullptr yTadShapeInfo by falling back to yShapeInfo,
    // and uses yTadNum==1 (yLen/xTadLen==1) to avoid dereferencing the null yTadOffsets.
    auto xTadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), dim, dimensionLength);
    auto xTadShapeInfo = xTadPack->specialShapeInfo();
    auto xOffsets = xTadPack->specialOffsets();

    // Check if all dimensions are valid for y's rank before creating y's TAD pack
    sd::LongType yRank = shape::rank(y->shapeInfo());
    bool yDimsValid = true;
    if (dim != nullptr) {
      for (sd::LongType i = 0; i < dimensionLength; i++) {
        sd::LongType d = dim[i];
        if (d < 0) d += yRank;
        if (d < 0 || d >= yRank) {
          yDimsValid = false;
          break;
        }
      }
    }

    const sd::LongType* yTadShapeInfo = nullptr;
    const sd::LongType* yTadOffsets = nullptr;
    std::shared_ptr<sd::TadPack> yTadPackHolder;  // keep alive until after the call
    if (yDimsValid) {
      yTadPackHolder = sd::ConstantTadHelper::getInstance().tadForDimensions(y->shapeInfo(), dim, dimensionLength);
      yTadShapeInfo = yTadPackHolder->specialShapeInfo();
      yTadOffsets = yTadPackHolder->specialOffsets();
    }

    auto lc = sd::LaunchContext::defaultContext();

    NativeOpExecutioner::execReduce3TAD(
        lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(), extraParams,
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->buffer(),
        y->shapeInfo(),
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->specialBuffer(),
        y->specialShapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo(),
        dim, dimensionLength,
        xTadShapeInfo, xOffsets, yTadShapeInfo, yTadOffsets);

    x->registerSpecialUse({z}, {x, y});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduce3Scalar(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray y, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x, y});

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduce3Scalar(
        lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special(), extraParams,
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->buffer(),
        y->shapeInfo(),
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->specialBuffer(),
        sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(y->shapeInfo())->special(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special());

    x->registerSpecialUse({z}, {x, y});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduce3All(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray z, OpaqueNDArray dimension, void *extraParams) {
  try {
    x->prepareSpecialUse({z}, {x, y, dimension});
    x->preparePrimaryUse({}, {dimension});

    auto dimensionPtr = dimension != nullptr ? reinterpret_cast<sd::LongType *>(dimension->buffer()) : nullptr;
    sd::LongType dimensionLength = static_cast<sd::LongType>(shape::length(dimension->shapeInfo()));

    auto xTadPack =
        sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), dimensionPtr, dimensionLength);
    auto xTadShapeInfo = xTadPack->specialShapeInfo();
    auto xOffsets = xTadPack->specialOffsets();

    auto yTadPack =
        sd::ConstantTadHelper::getInstance().tadForDimensions(y->shapeInfo(), dimensionPtr, dimensionLength);
    auto yTadShapeInfo = yTadPack->specialShapeInfo();
    auto yOffsets = yTadPack->specialOffsets();

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduce3All(lc, opNum,
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                        x->shapeInfo(),
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                        x->specialShapeInfo(),
                                        extraParams,
                                        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->buffer(),
                                        y->shapeInfo(),
                                        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->specialBuffer(),
                                        y->specialShapeInfo(),
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                        z->shapeInfo(),
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                        z->specialShapeInfo(),
                                        dimensionPtr,
                                        dimensionLength, xTadShapeInfo,
                                        xOffsets, yTadShapeInfo, yOffsets);

    x->registerSpecialUse({z}, {x, y});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
