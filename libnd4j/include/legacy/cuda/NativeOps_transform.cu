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

/**
 * Transform operations split from NativeOps.cu to reduce object file size
 * and prevent linker relocation overflow errors in large binaries.
 */

#include <cuda.h>
#include <system/Environment.h>
#include <legacy/NativeOpExecutioner.h>
#include <legacy/NativeOps.h>
#include <helpers/shape.h>
#include <execution/LaunchContext.h>

////////////////////////////////////////////////////////////////////////
void execTransformSame(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x});

    auto tadShapeInfo = reinterpret_cast<sd::LongType *>(extraPointers != nullptr ? extraPointers[0] : nullptr);
    auto tadOffsets = reinterpret_cast<sd::LongType *>(extraPointers != nullptr ? extraPointers[1] : nullptr);

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execTransformSame(lc, opNum,
                                           shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                           x->shapeInfo(),
                                           shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                           x->specialShapeInfo(),
                                           shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                           z->shapeInfo(),
                                           shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                           z->specialShapeInfo(),
                                           extraParams, tadShapeInfo, tadOffsets);

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execTransformBool(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x});

    auto tadShapeInfo = reinterpret_cast<sd::LongType *>(extraPointers != nullptr ? extraPointers[0] : nullptr);
    auto tadOffsets = reinterpret_cast<sd::LongType *>(extraPointers != nullptr ? extraPointers[1] : nullptr);

    bool xEmpty = shape::isEmptyConst(x->shapeInfo());
    bool zEmpty = shape::isEmptyConst(z->shapeInfo());

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execTransformBool(
        lc, opNum, xEmpty ? nullptr : x->buffer(), x->shapeInfo(),
        xEmpty ? nullptr : x->specialBuffer(), x->specialShapeInfo(),
        zEmpty ? nullptr : z->buffer(), z->shapeInfo(),
        zEmpty ? nullptr : z->specialBuffer(), z->specialShapeInfo(), extraParams);

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execTransformAny(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x});
    auto lc = sd::LaunchContext::defaultContext();

    NativeOpExecutioner::execTransformAny(
        lc, opNum, shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(), x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(), x->specialShapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(), z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(), z->specialShapeInfo(), extraParams, false);

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execTransformStrict(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x});

    auto tadShapeInfo = reinterpret_cast<sd::LongType *>(extraPointers != nullptr ? extraPointers[10] : nullptr);
    auto tadOffsets = reinterpret_cast<sd::LongType *>(extraPointers != nullptr ? extraPointers[11] : nullptr);

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execTransformStrict(
        lc, opNum, shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(), x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(), x->specialShapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(), z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(), z->specialShapeInfo(), extraParams);

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execTransformFloat(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x});

    auto tadShapeInfo = reinterpret_cast<sd::LongType *>(extraPointers != nullptr ? extraPointers[10] : nullptr);
    auto tadOffsets = reinterpret_cast<sd::LongType *>(extraPointers != nullptr ? extraPointers[11] : nullptr);

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execTransformFloat(
        lc, opNum, shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(), x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(), x->specialShapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(), z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(), z->specialShapeInfo(), extraParams);

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
