/* ******************************************************************************
 * Copyright (c) 2015-2026 The Eclipse Foundation.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <execution/LaunchContext.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>
#include <helpers/shape.h>
#include <legacy/NativeOpExecutioner.h>
#include <legacy/NativeOps.h>
#include <system/Environment.h>


void execPairwiseTransform(sd::Pointer *extraPointers, int opNum,
                           OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray z,
                           void *extraParams) {
  try {
    x->prepareSpecialUse({z}, {x, y});

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execPairwiseTransform(
        lc, opNum,
        sd::LegacyTensorArg::fromArray(x),
        sd::LegacyTensorArg::fromArray(y),
        sd::LegacyTensorArg::fromArray(z), extraParams);

    x->registerSpecialUse({z}, {x, y});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(
        e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execPairwiseTransformBool(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x, y});

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execPairwiseBoolTransform(
        lc, opNum,
        sd::LegacyTensorArg::fromArray(x),
        sd::LegacyTensorArg::fromArray(y),
        sd::LegacyTensorArg::fromArray(z),
        extraParams);

    x->registerSpecialUse({z}, {x, y});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execSummaryStatsScalar(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, bool biasCorrected) {
  try {
    x->prepareSpecialUse({z}, {x});

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execSummaryStatsScalar(
        lc, opNum,
        sd::LegacyTensorArg::fromArray(x),

        extraParams,
        sd::LegacyTensorArg::fromArray(z),
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
        sd::LegacyTensorArg::withShape(
            x, x->shapeInfo(), sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special()),
        extraParams,
        sd::LegacyTensorArg::withShape(
            z, z->shapeInfo(), sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special()),
        dimensionData, dimensionLength, tadShapeInfo, tadOffsets, biasCorrected);

    x->registerSpecialUse({z}, {x});
    dimension->registerSpecialUse({}, {dimension});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execBroadcastBool(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray z, void *extraParams, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x, y, dimension});

    // Axes are host-side semantic metadata; tensor operands remain device-owned.
    if (dimension != nullptr) dimension->preparePrimaryUse({}, {dimension});
    auto dimensionBuffer = dimension != nullptr ? reinterpret_cast<sd::LongType *>(dimension->buffer()) : nullptr;
    sd::LongType dimensionLength = dimension != nullptr ? shape::length(dimension->shapeInfo()) : 0;

    auto tadOnlyShapeInfo = extraPointers != nullptr ? reinterpret_cast<sd::LongType *>(extraPointers[10]) : nullptr;
    auto tadOffsets = extraPointers != nullptr ? reinterpret_cast<sd::LongType *>(extraPointers[11]) : nullptr;
    auto tadOnlyShapeInfoZ = extraPointers != nullptr ? reinterpret_cast<sd::LongType *>(extraPointers[12]) : nullptr;
    auto tadOffsetsZ = extraPointers != nullptr ? reinterpret_cast<sd::LongType *>(extraPointers[13]) : nullptr;

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execBroadcastBool(
        lc, opNum,
        sd::LegacyTensorArg::fromArray(x),
        sd::LegacyTensorArg::fromArray(y),
        sd::LegacyTensorArg::fromArray(z),
        extraParams,
        dimensionBuffer,
        dimensionLength,
        tadOnlyShapeInfo,
        tadOffsets,
        tadOnlyShapeInfoZ,
        tadOffsetsZ);

    x->registerSpecialUse({z}, {x, y, dimension});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execBroadcast(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray z, void *extraParams, OpaqueNDArray dimension) {
  try {
    if (extraParams != nullptr) {
      THROW_EXCEPTION("Vulkan legacy broadcast cannot infer the count of extra parameters");
    }
    x->prepareSpecialUse({z}, {x, y, dimension});

    // Axes are host-side semantic metadata; tensor operands remain device-owned.
    if (dimension != nullptr) dimension->preparePrimaryUse({}, {dimension});
    auto dimensionBuffer = dimension != nullptr ? reinterpret_cast<sd::LongType *>(dimension->buffer()) : nullptr;
    sd::LongType dimensionLength = dimension != nullptr ? shape::length(dimension->shapeInfo()) : 0;

    auto tadOnlyShapeInfo = extraPointers != nullptr ? reinterpret_cast<sd::LongType *>(extraPointers[10]) : nullptr;
    auto tadOffsets = extraPointers != nullptr ? reinterpret_cast<sd::LongType *>(extraPointers[11]) : nullptr;
    auto tadOnlyShapeInfoZ = extraPointers != nullptr ? reinterpret_cast<sd::LongType *>(extraPointers[12]) : nullptr;
    auto tadOffsetsZ = extraPointers != nullptr ? reinterpret_cast<sd::LongType *>(extraPointers[13]) : nullptr;

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execBroadcast(
        lc, opNum,
        sd::LegacyTensorArg::fromArray(x),
        sd::LegacyTensorArg::fromArray(y),
        sd::LegacyTensorArg::fromArray(z),
        dimensionBuffer,
        dimensionLength,
        tadOnlyShapeInfo,
        tadOffsets,
        tadOnlyShapeInfoZ,
        tadOffsetsZ);

    x->registerSpecialUse({z}, {x, y, dimension});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execScalarBool(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, void *extraParams) {
  try {
    x->prepareSpecialUse({z}, {x, scalar});

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execScalarBool(
        lc, opNum,
        sd::LegacyTensorArg::withShape(
            x, x->shapeInfo(), sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special()),
        sd::LegacyTensorArg::withShape(
            z, z->shapeInfo(), sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special()),
        sd::LegacyTensorArg::withShape(
            scalar, scalar->shapeInfo(), sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(scalar->shapeInfo())->special()), extraParams);

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
        sd::LegacyTensorArg::withShape(
            x, x->shapeInfo(), sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special()),
        extraParams,
        sd::LegacyTensorArg::withShape(
            z, z->shapeInfo(), sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special()),
        sd::LegacyTensorArg::withShape(
            scalar, scalar->shapeInfo(), sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(scalar->shapeInfo())->special()),

        dim, dimensionLength,
        xTadShapeInfo, xOffsets, zTadShapeInfo, zOffsets);

    x->registerSpecialUse({z}, {x, scalar});
    dimension->registerSpecialUse({}, {dimension});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execScalar(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, void *extraParams) {
  try {
    x->prepareSpecialUse({z}, {x, scalar});

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execScalar(
        lc, opNum,
        sd::LegacyTensorArg::withShape(
            x, x->shapeInfo(), sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special()),
        sd::LegacyTensorArg::withShape(
            z, z->shapeInfo(), sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special()),
        sd::LegacyTensorArg::withShape(
            scalar, scalar->shapeInfo(), sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(scalar->shapeInfo())->special()), extraParams);

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
    auto zTadPack =
        sd::ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(), dimensionPtr, dimensionLength);

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execScalar(
        lc, opNum,
        sd::LegacyTensorArg::fromArray(x),
        extraParams,
        sd::LegacyTensorArg::fromArray(z),
        sd::LegacyTensorArg::fromArray(scalar),
        dimensionPtr, dimensionLength,
        xTadPack->specialShapeInfo(), xTadPack->specialOffsets(),
        zTadPack->specialShapeInfo(), zTadPack->specialOffsets());

    x->registerSpecialUse({z}, {x, scalar});
    dimension->registerSpecialUse({}, {dimension});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceFloat(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x});

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduceFloatScalar(
        lc, opNum,
        sd::LegacyTensorArg::fromArray(x),
        extraParams,
        sd::LegacyTensorArg::fromArray(z));

    x->registerSpecialUse({z}, {x});
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
    const sd::LongType *zShapeInfoD = z->specialShapeInfo();

    if (!isFullArrayReduce && shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> *dims =
        isFullArrayReduce ? new std::vector<sd::LongType>() : new std::vector<sd::LongType>(dimensions);

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduceFloat(lc,
                                         opNum,
                                         sd::LegacyTensorArg::fromArray(x), extraParams,
                                         sd::LegacyTensorArg::withShape(z, zShapeInfoH, zShapeInfoD),
                                         dims->data(), dims->size());

    x->registerSpecialUse({z}, {x});
    delete dims;
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
        sd::LegacyTensorArg::fromArray(x), extraParams,
        sd::LegacyTensorArg::fromArray(z));

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
    const sd::LongType *zShapeInfoD = z->specialShapeInfo();

    if (!isFullArrayReduce && shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> *dims =
        isFullArrayReduce ? new std::vector<sd::LongType>() : new std::vector<sd::LongType>(dimensions);

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduceSame(lc,
                                        opNum,
                                        sd::LegacyTensorArg::fromArray(x), extraParams,
                                        sd::LegacyTensorArg::withShape(z, zShapeInfoH, zShapeInfoD),
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
    const sd::LongType *zShapeInfoD = z->specialShapeInfo();

    if (shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> *dims =
        (dimensionLength == 1 && (dimensions[0] == -1 || dimensions[0] == SD_MAX_INT))
            ? new std::vector<sd::LongType>() : new std::vector<sd::LongType>(dimensions);

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduceLong(lc, opNum,
                                        sd::LegacyTensorArg::fromArray(x), extraParams,
                                        sd::LegacyTensorArg::withShape(z, zShapeInfoH, zShapeInfoD),
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
    const sd::LongType *zShapeInfoD = z->specialShapeInfo();

    if (shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> *dims =
        (dimensionLength == 1 && (dimensions[0] == -1 || dimensions[0] == SD_MAX_INT))
            ? new std::vector<sd::LongType>() : new std::vector<sd::LongType>(dimensions);

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduceBool(lc,
                                        opNum,
                                        sd::LegacyTensorArg::fromArray(x), extraParams,
                                        sd::LegacyTensorArg::withShape(z, zShapeInfoH, zShapeInfoD),
                                        dims->data(), dims->size());

    x->registerSpecialUse({z}, {x});

    delete dims;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceLong2(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,
                     void *extraParams, OpaqueNDArray z,
                     OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData =
        dimension != nullptr
            ? reinterpret_cast<sd::LongType *>(dimension->buffer())
            : nullptr;
    sd::LongType dimensionLength =
        static_cast<sd::LongType>(shape::length(dimension->shapeInfo()));

    const auto zLen = shape::length(z->shapeInfo());

    std::vector<sd::LongType> dimensions(dimensionData,
                                         dimensionData + dimensionLength);

    bool isFullArrayReduce =
        (dimensionLength == 1 &&
         (dimensions[0] == -1 || dimensions[0] == SD_MAX_INT));

    const sd::LongType *zShapeInfoH = z->shapeInfo();
    const sd::LongType *zShapeInfoD = z->specialShapeInfo();

    if (!isFullArrayReduce &&
        shape::rank(x->shapeInfo()) - dimensionLength !=
            shape::rank(z->shapeInfo()) &&
        zLen != 1) {
      auto zPack = sd::ConstantShapeHelper::getInstance()
                       .createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(),
                                                              &dimensions);
      zShapeInfoH =
          reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD =
          reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> *dims =
        isFullArrayReduce ? new std::vector<sd::LongType>() : new std::vector<sd::LongType>(dimensions);

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduceLong(
        lc, opNum,
        sd::LegacyTensorArg::fromArray(x), extraParams,
        sd::LegacyTensorArg::withShape(z, zShapeInfoH, zShapeInfoD), dims->data(), dims->size());

    x->registerSpecialUse({z}, {x});

    delete dims;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(
        e.what());
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
    const sd::LongType *zShapeInfoD = z->specialShapeInfo();

    if (!isFullArrayReduce && shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> *dims =
        isFullArrayReduce ? new std::vector<sd::LongType>() : new std::vector<sd::LongType>(dimensions);

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduceBool(lc,
                                        opNum,
                                        sd::LegacyTensorArg::fromArray(x), extraParams,
                                        sd::LegacyTensorArg::withShape(z, zShapeInfoH, zShapeInfoD),
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
        sd::LegacyTensorArg::fromArray(x), extraParams,
        sd::LegacyTensorArg::fromArray(z),
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
        sd::LegacyTensorArg::fromArray(x), extraParams,
        sd::LegacyTensorArg::fromArray(z));

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduce3(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray y, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x, y});

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execReduce3(
        lc,
        opNum,
        sd::LegacyTensorArg::fromArray(x),
        extraParams,
        sd::LegacyTensorArg::fromArray(y),
        sd::LegacyTensorArg::fromArray(z));

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
        sd::LegacyTensorArg::withShape(
            x, x->shapeInfo(), sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special()), extraParams,
        sd::LegacyTensorArg::withShape(
            y, y->shapeInfo(), sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(y->shapeInfo())->special()),
        sd::LegacyTensorArg::withShape(
            z, z->shapeInfo(), sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special()));

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
    // When y has a lower rank than x (for example, a rank-1 needle versus rank-2 x),
    // the dimensions used to TAD x are out of bounds for y, so omit y's derived TAD metadata.
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
        sd::LegacyTensorArg::fromArray(x), extraParams,
        sd::LegacyTensorArg::fromArray(y),
        sd::LegacyTensorArg::fromArray(z),
        dim, dimensionLength,
        xTadShapeInfo, xOffsets, yTadShapeInfo, yTadOffsets);

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
                                        sd::LegacyTensorArg::fromArray(x),
                                        extraParams,
                                        sd::LegacyTensorArg::fromArray(y),
                                        sd::LegacyTensorArg::fromArray(z),
                                        dimensionPtr,
                                        dimensionLength, xTadShapeInfo,
                                        xOffsets, yTadShapeInfo, yOffsets);

    x->registerSpecialUse({z}, {x, y});
  } catch (std::exception &e) {

    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execTransformSame(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x});

    auto tadShapeInfo = reinterpret_cast<sd::LongType *>(extraPointers != nullptr ? extraPointers[0] : nullptr);
    auto tadOffsets = reinterpret_cast<sd::LongType *>(extraPointers != nullptr ? extraPointers[1] : nullptr);

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execTransformSame(lc, opNum,
                                           sd::LegacyTensorArg::fromArray(x),
                                           sd::LegacyTensorArg::fromArray(z),
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

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execTransformBool(
        lc, opNum, sd::LegacyTensorArg::fromArray(x), sd::LegacyTensorArg::fromArray(z), extraParams);

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
        lc, opNum, sd::LegacyTensorArg::fromArray(x), sd::LegacyTensorArg::fromArray(z), extraParams, false);

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
        lc, opNum, sd::LegacyTensorArg::fromArray(x), sd::LegacyTensorArg::fromArray(z), extraParams);

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
        lc, opNum, sd::LegacyTensorArg::fromArray(x), sd::LegacyTensorArg::fromArray(z), extraParams);

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

