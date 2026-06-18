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
// Contains: execSummaryStats, execSummaryStatsTad
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
void execSummaryStats(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, bool biasCorrected) {
  try {
    x->prepareSpecialUse({z}, {x});

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execSummaryStats(lc,
                                          opNum,
                                          shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                          x->shapeInfo(),
                                          shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                          sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special(),
                                          extraParams,
                                          shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                          z->shapeInfo(),
                                          shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                          sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special(),
                                          biasCorrected);

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execSummaryStatsTad(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z,
                         OpaqueNDArray dimension, bool biasCorrected) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData = dimension != nullptr ? reinterpret_cast<sd::LongType *>(dimension->buffer()) : nullptr;
    int dimensionLength = static_cast<int>(shape::length(dimension->shapeInfo()));

    auto tadPack =
        sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), dimensionData, dimensionLength);
    auto tadShapeInfo = tadPack->specialShapeInfo();
    auto tadOffsets = tadPack->specialOffsets();

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execSummaryStats(
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
        dimensionData, dimensionLength, tadShapeInfo, tadOffsets, biasCorrected);

    x->registerSpecialUse({z}, {x});
    dimension->registerSpecialUse({}, {dimension});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
