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
 * Reduce operations split from NativeOps.cu to reduce object file size
 * and prevent linker relocation overflow errors in large binaries.
 */

#include <cuda.h>
#include <system/Environment.h>
#include <legacy/NativeOpExecutioner.h>
#include <legacy/NativeOps.h>
#include <helpers/shape.h>
#include <helpers/ShapeUtils.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <execution/LaunchContext.h>

////////////////////////////////////////////////////////////////////////
void execReduceFloat(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x});

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduceFloatScalar(
        lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(),
        extraParams,
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo());

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceSame(sd::Pointer *extraPointers,
                    int opNum,
                    OpaqueNDArray x,
                    void *extraParams,
                    OpaqueNDArray z) {
  try {
    if (x == nullptr) {
      THROW_EXCEPTION("execReduceSame: input array x is null");
    }
    if (z == nullptr) {
      THROW_EXCEPTION("execReduceSame: output array z is null");
    }

    x->prepareSpecialUse({z}, {x});

    bool xIsEmpty = shape::isEmptyConst(x->shapeInfo());
    bool zIsEmpty = shape::isEmptyConst(z->shapeInfo());

    if (!xIsEmpty && x->specialBuffer() == nullptr) {
      std::string msg = "execReduceSame: input array x has no device buffer after sync. "
                        "Rank: " + std::to_string(shape::rank(x->shapeInfo())) +
                        ", length: " + std::to_string(shape::length(x->shapeInfo()));
      THROW_EXCEPTION(msg.c_str());
    }
    if (!zIsEmpty && z->specialBuffer() == nullptr) {
      std::string msg = "execReduceSame: output array z has no device buffer after sync. "
                        "Rank: " + std::to_string(shape::rank(z->shapeInfo())) +
                        ", length: " + std::to_string(shape::length(z->shapeInfo()));
      THROW_EXCEPTION(msg.c_str());
    }

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduceSameScalar(
        lc, opNum,
        xIsEmpty ? nullptr : x->buffer(),
        x->shapeInfo(),
        xIsEmpty ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(), extraParams,
        z->buffer(),
        z->shapeInfo(),
        zIsEmpty ? nullptr : z->specialBuffer(),
        z->specialShapeInfo());

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceSame2(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData = dimension != nullptr ? reinterpret_cast<sd::LongType *>(dimension->buffer()) : nullptr;
    sd::LongType dimensionLength = static_cast<sd::LongType>(shape::length(dimension->shapeInfo()));

    const auto zLen = shape::length(z->shapeInfo());

    std::vector<sd::LongType> dimensions(dimensionData, dimensionData + dimensionLength);

    bool isFullArrayReduce = (dimensionLength == 1 && (dimensions[0] == -1 || dimensions[0] == SD_MAX_INT));

    const sd::LongType *zShapeInfoH = z->shapeInfo();

    if (!isFullArrayReduce && shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
    }

    std::vector<sd::LongType> *dims =
        (zLen != 1 && !isFullArrayReduce) ? sd::ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) : new std::vector<sd::LongType>();

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduceSame(lc,
                                        opNum,
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                        x->shapeInfo(),
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                        x->specialShapeInfo(), extraParams,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                        zShapeInfoH,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                        z->specialShapeInfo(),
                                        dims->data(), dims->size());

    x->registerSpecialUse({z}, {x});

    delete dims;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceLong2(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData = dimension != nullptr ? reinterpret_cast<sd::LongType *>(dimension->buffer()) : nullptr;
    sd::LongType dimensionLength = static_cast<sd::LongType>(shape::length(dimension->shapeInfo()));

    const auto zLen = shape::length(z->shapeInfo());

    std::vector<sd::LongType> dimensions(dimensionData, dimensionData + dimensionLength);

    bool isFullArrayReduce = (dimensionLength == 1 && (dimensions[0] == -1 || dimensions[0] == SD_MAX_INT));

    const sd::LongType *zShapeInfoH = z->shapeInfo();

    if (!isFullArrayReduce && shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
    }

    std::vector<sd::LongType> *dims =
        (zLen != 1 && !isFullArrayReduce) ? sd::ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) : new std::vector<sd::LongType>();

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduceLong(lc, opNum,
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                        x->shapeInfo(),
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                        x->specialShapeInfo(), extraParams,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                        zShapeInfoH,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                        z->specialShapeInfo(),
                                        dims->data(), dims->size());

    x->registerSpecialUse({z}, {x});

    delete dims;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceLong(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData = dimension != nullptr ? reinterpret_cast<sd::LongType *>(dimension->buffer()) : nullptr;
    sd::LongType dimensionLength = static_cast<sd::LongType>(shape::length(dimension->shapeInfo()));

    const auto zLen = shape::length(z->shapeInfo());

    std::vector<sd::LongType> dimensions(dimensionData, dimensionData + dimensionLength);

    const sd::LongType *zShapeInfoH = z->shapeInfo();

    if (shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
    }

    std::vector<sd::LongType> *dims =
        (zLen != 1) ? sd::ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) : new std::vector<sd::LongType>();

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduceLong(lc, opNum,
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                        x->shapeInfo(),
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                        x->specialShapeInfo(), extraParams,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                        zShapeInfoH,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                        z->specialShapeInfo(),
                                        dims->data(), dims->size());

    x->registerSpecialUse({z}, {x});

    delete dims;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceBool2(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData = dimension != nullptr ? reinterpret_cast<sd::LongType *>(dimension->buffer()) : nullptr;
    sd::LongType dimensionLength = static_cast<sd::LongType>(shape::length(dimension->shapeInfo()));

    const auto zLen = shape::length(z->shapeInfo());

    std::vector<sd::LongType> dimensions(dimensionData, dimensionData + dimensionLength);

    bool isFullArrayReduce = (dimensionLength == 1 && (dimensions[0] == -1 || dimensions[0] == SD_MAX_INT));

    const sd::LongType *zShapeInfoH = z->shapeInfo();

    if (!isFullArrayReduce && shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
    }

    std::vector<sd::LongType> *dims =
        (zLen != 1 && !isFullArrayReduce) ? sd::ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) : new std::vector<sd::LongType>();

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduceBool(lc,
                                        opNum,
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                        x->shapeInfo(),
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                        x->specialShapeInfo(), extraParams,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                        zShapeInfoH,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                        z->specialShapeInfo(),
                                        dims->data(), dims->size());

    x->registerSpecialUse({z}, {x});

    delete dims;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceBool(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData = dimension != nullptr ? reinterpret_cast<sd::LongType *>(dimension->buffer()) : nullptr;
    sd::LongType dimensionLength = static_cast<sd::LongType>(shape::length(dimension->shapeInfo()));

    const auto zLen = shape::length(z->shapeInfo());

    std::vector<sd::LongType> dimensions(dimensionData, dimensionData + dimensionLength);

    const sd::LongType *zShapeInfoH = z->shapeInfo();

    if (shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
    }

    std::vector<sd::LongType> *dims =
        (zLen != 1) ? sd::ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) : new std::vector<sd::LongType>();

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduceBool(lc,
                                        opNum,
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                        x->shapeInfo(),
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                        x->specialShapeInfo(), extraParams,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                        zShapeInfoH,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                        z->specialShapeInfo(),
                                        dims->data(), dims->size());

    x->registerSpecialUse({z}, {x});

    delete dims;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceFloat2(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData = dimension != nullptr ? reinterpret_cast<sd::LongType *>(dimension->buffer()) : nullptr;
    sd::LongType dimensionLength = static_cast<sd::LongType>(shape::length(dimension->shapeInfo()));

    const auto zLen = shape::length(z->shapeInfo());

    std::vector<sd::LongType> dimensions(dimensionData, dimensionData + dimensionLength);

    bool isFullArrayReduce = (dimensionLength == 1 && (dimensions[0] == -1 || dimensions[0] == SD_MAX_INT));

    const sd::LongType *zShapeInfoH = z->shapeInfo();

    if (!isFullArrayReduce && shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
    }

    std::vector<sd::LongType> *dims =
        (zLen != 1 && !isFullArrayReduce) ? sd::ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) : new std::vector<sd::LongType>();

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduceFloat(lc,
                                         opNum,
                                         shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                         x->shapeInfo(),
                                         shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                         x->specialShapeInfo(), extraParams,
                                         shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                         zShapeInfoH,
                                         shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                         z->specialShapeInfo(),
                                         dims->data(), dims->size());

    x->registerSpecialUse({z}, {x});
    delete dims;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execIndexReduce(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData = dimension != nullptr ? reinterpret_cast<sd::LongType *>(dimension->buffer()) : nullptr;
    sd::LongType dimensionLength = static_cast<sd::LongType>(shape::length(dimension->shapeInfo()));

    auto tadPack =
        sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), dimensionData, dimensionLength);

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execIndexReduce(
        lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(), extraParams,
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo(),
        dimensionData, dimensionLength, tadPack->specialShapeInfo(), tadPack->specialOffsets());

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execIndexReduceScalar(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x});

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execIndexReduceScalar(
        lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(), extraParams,
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo());

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
