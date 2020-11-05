/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by agibsonccc on 2/21/16.
//

#define __STDC_CONSTANT_MACROS

#include <legacy/NativeOps.h>
#include "legacy/NativeOpExecutioner.h"
#include <array/NDArray.h>
#include <graph/GraphExecutioner.h>
#include <graph/GraphHolder.h>
#include <math/templatemath.h>
#include <types/float8.h>
#include <loops/type_conversions.h>
#include <helpers/helper_ptrmap.h>
#include <helpers/logger.h>
#include <system/pointercast.h>
#include <system/pairwise_util.h>
#include <types/types.h>
#include <ops/declarable/helpers/transforms.h>
#include <exceptions/allocation_exception.h>
#include <helpers/BlasHelper.h>


#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#else
#include <io.h>
#include <helpers/mman.h>
#endif
#include <sys/types.h>

#include <ops/declarable/CustomOperations.h>
#include <errno.h>


char *name;
bool nameSet = false;


#ifdef __ND4J_EXPERIMENTAL__
bool experimentalSupport = true;
#else
bool experimentalSupport = false;
#endif

#include <ops/specials.h>
#include <system/Environment.h>
#include <helpers/TAD.h>
#include <ops/declarable/OpRegistrator.h>
#include <graph/Context.h>
#include <graph/ResultWrapper.h>
#include <helpers/DebugHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <performance/benchmarking/BenchmarkSuit.h>
#include <performance/benchmarking/FullBenchmarkSuit.h>
#include <performance/benchmarking/LightBenchmarkSuit.h>
#include <execution/Threads.h>

#ifdef CPU_FEATURES
#include <cpuinfo_x86.h>
#endif

using namespace sd;

void setElementThreshold(int num) {
    if (num > 0)
        sd::Environment::getInstance().setElementwiseThreshold(num);
}

void setTADThreshold(int num) {
    if (num > 0)
        sd::Environment::getInstance().setTadThreshold(num);
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 */
void execIndexReduceScalar(Nd4jPointer *extraPointers,
                                                int opNum,
                                                OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                                                void *extraParams,
                                                OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo) {
    try {
        NativeOpExecutioner::execIndexReduceScalar(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo, extraParams, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @param hZ
 * @param hZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void  execIndexReduce(Nd4jPointer *extraPointers,int opNum,
                                        OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                                        void *extraParams,
                                        OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
                                        OpaqueDataBuffer *dbDimension, const Nd4jLong *hDimensionShape, const Nd4jLong *dDimensionShape) {
    try {
        auto dimension = reinterpret_cast<int *>(dbDimension->primary());
        int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

        auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension,
                                                                                dimensionLength);

        auto hTADShapeInfo = tadPack.primaryShapeInfo();
        auto hTADOffsets = tadPack.primaryOffsets();

        auto hz = reinterpret_cast<Nd4jLong *>(dbZ->primary());

        NativeOpExecutioner::execIndexReduce(nullptr, opNum,
                                             dbX->primary(),
                                             hXShapeInfo,
                                             dbX->special(),
                                             dXShapeInfo,
                                             extraParams,
                                             hz,
                                             hZShapeInfo,
                                             dbZ->special(),
                                             dZShapeInfo,
                                             dimension,
                                             dimensionLength,
                                             hTADShapeInfo,
                                             hTADOffsets);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}


/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param hY
 * @param hYShapeInfo
 * @param hZ
 * @param hZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void execBroadcast(Nd4jPointer *extraPointers,
                                      int opNum,
                                      OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                                      OpaqueDataBuffer *dbY, const Nd4jLong *hYShapeInfo, const Nd4jLong *dYShapeInfo,
                                      OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
                                      OpaqueDataBuffer *dbDimension, const Nd4jLong *hDimensionShape, const Nd4jLong *dDimensionShape) {
    try {
        auto dimension = reinterpret_cast<int *>(dbDimension->primary());
        auto dimensionLength = static_cast<int>(shape::length(hDimensionShape));

        auto tadPackX = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);
        auto tadPackZ = sd::ConstantTadHelper::getInstance().tadForDimensions(hZShapeInfo, dimension, dimensionLength);

        auto hTADShapeInfo = tadPackX.primaryShapeInfo();
        auto hTADOffsets = tadPackX.primaryOffsets();
        auto hTADShapeInfoZ = tadPackZ.primaryShapeInfo();
        auto hTADOffsetsZ = tadPackZ.primaryOffsets();

        NativeOpExecutioner::execBroadcast(nullptr,
                                           opNum,
                                           dbX->primary(),
                                           hXShapeInfo,
                                           dbX->special(),
                                           dXShapeInfo,
                                           dbY->primary(),
                                           hYShapeInfo,
                                           dbY->special(),
                                           dYShapeInfo,
                                           dbZ->primary(), hZShapeInfo,
                                           dbZ->special(), dZShapeInfo,
                                           dimension,
                                           dimensionLength, hTADShapeInfo, hTADOffsets, hTADShapeInfoZ, hTADOffsetsZ);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execBroadcastBool(Nd4jPointer *extraPointers,
                              int opNum,
                              OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                              OpaqueDataBuffer *dbY, const Nd4jLong *hYShapeInfo, const Nd4jLong *dYShapeInfo,
                              OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
                              void *extraParams,
                              OpaqueDataBuffer *dbDimension, const Nd4jLong *hDimensionShape, const Nd4jLong *dDimensionShape) {
    try {
        auto dimension = reinterpret_cast<int *>(dbDimension->primary());
        auto dimensionLength = static_cast<int>(shape::length(hDimensionShape));

        auto tadPackX = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);
        auto tadPackZ = sd::ConstantTadHelper::getInstance().tadForDimensions(hZShapeInfo, dimension, dimensionLength);

        auto hTADShapeInfo = tadPackX.primaryShapeInfo();
        auto hTADOffsets = tadPackX.primaryOffsets();
        auto hTADShapeInfoZ = tadPackZ.primaryShapeInfo();
        auto hTADOffsetsZ = tadPackZ.primaryOffsets();

        NativeOpExecutioner::execBroadcastBool(nullptr,
                                               opNum,
                                               dbX->primary(),
                                               hXShapeInfo,
                                               dbX->special(),
                                               dXShapeInfo,
                                               dbY->primary(),
                                               hYShapeInfo,
                                               dbY->special(),
                                               dYShapeInfo,
                                               dbZ->primary(), hZShapeInfo,
                                               dbZ->special(), dZShapeInfo,
                                               extraParams,
                                               dimension,
                                               dimensionLength, hTADShapeInfo, hTADOffsets, hTADShapeInfoZ,
                                               hTADOffsetsZ);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param hY
 * @param hYShapeInfo
 * @param hZ
 * @param hZShapeInfo
 * @param extraParams
 * @param n
 */
void execPairwiseTransform(
        Nd4jPointer *extraPointers,
        int opNum,
        OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
        OpaqueDataBuffer *dbY, const Nd4jLong *hYShapeInfo, const Nd4jLong *dYShapeInfo,
        OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
        void *extraParams) {
    try {
        NativeOpExecutioner::execPairwiseTransform(nullptr,
                                                   opNum,
                                                   dbX->primary(),
                                                   hXShapeInfo,
                                                   dbX->special(),
                                                   dXShapeInfo,
                                                   dbY->primary(),
                                                   hYShapeInfo,
                                                   dbY->special(),
                                                   dYShapeInfo,
                                                   dbZ->primary(),
                                                   hZShapeInfo,
                                                   dbZ->special(),
                                                   dZShapeInfo,
                                                   extraParams);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execPairwiseTransformBool(
        Nd4jPointer *extraPointers,
        int opNum,
        OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
        OpaqueDataBuffer *dbY, const Nd4jLong *hYShapeInfo, const Nd4jLong *dYShapeInfo,
        OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
        void *extraParams) {

    try {
        NativeOpExecutioner::execPairwiseBoolTransform(nullptr,
                                                       opNum,
                                                       dbX->primary(),
                                                       hXShapeInfo,
                                                       dbX->special(),
                                                       dXShapeInfo,
                                                       dbY->primary(),
                                                       hYShapeInfo,
                                                       dbY->special(),
                                                       dYShapeInfo,
                                                       dbZ->primary(),
                                                       hZShapeInfo,
                                                       dbZ->special(),
                                                       dZShapeInfo,
                                                       extraParams);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @param hZ
 * @param hZShapeInfo
 */
void execReduceFloat(
        Nd4jPointer *extraPointers,
        int opNum,
        OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
        void *extraParams,
        OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo) {

    try {
        NativeOpExecutioner::execReduceFloatScalar(nullptr,
                                                   opNum,
                                                   dbX->primary(),
                                                   hXShapeInfo,
                                                   dbX->special(),
                                                   dXShapeInfo,
                                                   extraParams,
                                                   dbZ->primary(),
                                                   hZShapeInfo,
                                                   dbZ->special(),
                                                   dZShapeInfo);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execReduceSame(
        Nd4jPointer *extraPointers,
        int opNum,
        OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
        void *extraParams,
        OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo) {

    try {
        NativeOpExecutioner::execReduceSameScalar(nullptr,
                                                  opNum,
                                                  dbX->primary(),
                                                  hXShapeInfo,
                                                  dbX->special(),
                                                  dXShapeInfo,
                                                  extraParams,
                                                  dbZ->primary(),
                                                  hZShapeInfo,
                                                  dbZ->special(),
                                                  dZShapeInfo);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execReduceBool(
        Nd4jPointer *extraPointers,
        int opNum,
        OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
        void *extraParams,
        OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo) {
    try {
        NativeOpExecutioner::execReduceBoolScalar(nullptr,
                                                  opNum,
                                                  dbX->primary(),
                                                  hXShapeInfo,
                                                  dbX->special(),
                                                  dXShapeInfo,
                                                  extraParams,
                                                  dbZ->primary(),
                                                  hZShapeInfo,
                                                  dbZ->special(),
                                                  dZShapeInfo);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execReduceLong(
        Nd4jPointer *extraPointers,
        int opNum,
        OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
        void *extraParams,
        OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo) {
    try {
        NativeOpExecutioner::execReduceLongScalar(nullptr,
                                                  opNum,
                                                  dbX->primary(),
                                                  hXShapeInfo,
                                                  dbX->special(),
                                                  dXShapeInfo,
                                                  extraParams,
                                                  dbZ->primary(),
                                                  hZShapeInfo,
                                                  dbZ->special(),
                                                  dZShapeInfo);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @param hZ
 * @param hZShapeInfo
 */
void execReduceFloat2(Nd4jPointer *extraPointers,
                                   int opNum,
                                   OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                                   void *extraParams,
                                   OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
                                   OpaqueDataBuffer *dbDimension, const Nd4jLong *hDimensionShape, const Nd4jLong *dDimensionShape) {
    try {

        auto dimension = reinterpret_cast<int *>(dbDimension->primary());
        auto dimensionLength = static_cast<int>(shape::length(hDimensionShape));

        const auto zLen = shape::length(hZShapeInfo);

        std::vector<int> dimensions(dimension, dimension + dimensionLength);

        const Nd4jLong* zShapeInfoH = hZShapeInfo;
        const Nd4jLong* zShapeInfoD = dZShapeInfo;

        if(shape::rank(hXShapeInfo) - dimensionLength != shape::rank(hZShapeInfo) && zLen != 1) {
            auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(hZShapeInfo, dimensions);
            zShapeInfoH = reinterpret_cast<Nd4jLong const*>(zPack.primary());
            zShapeInfoD = reinterpret_cast<Nd4jLong const*>(zPack.special());
        }

        std::vector<int> dims = (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(hXShapeInfo), dimensions) : std::vector<int>();
        NativeOpExecutioner::execReduceFloat(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo, extraParams, dbZ->primary(), zShapeInfoH, dbZ->special(), zShapeInfoD, dims.data(), dims.size());

    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execReduceBool2(Nd4jPointer *extraPointers,
                                int opNum,
                                OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                                void *extraParams,
                                OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
                                OpaqueDataBuffer *dbDimension, const Nd4jLong *hDimensionShape, const Nd4jLong *dDimensionShape) {

    try {
        auto dimension = reinterpret_cast<int *>(dbDimension->primary());
        auto dimensionLength = static_cast<int>(shape::length(hDimensionShape));

        std::vector<int> dimensions(dimension, dimension + dimensionLength);

        const auto zLen = shape::length(hZShapeInfo);

        const Nd4jLong* zShapeInfoH = hZShapeInfo;
        const Nd4jLong* zShapeInfoD = dZShapeInfo;

        if(shape::rank(hXShapeInfo) - dimensionLength != shape::rank(hZShapeInfo)) {
            auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(hZShapeInfo, dimensions);
            zShapeInfoH = reinterpret_cast<Nd4jLong const*>(zPack.primary());
            zShapeInfoD = reinterpret_cast<Nd4jLong const*>(zPack.special());
        }

        std::vector<int> dims = (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(hXShapeInfo), dimensions) : std::vector<int>();
        NativeOpExecutioner::execReduceBool(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo, extraParams, dbZ->primary(), zShapeInfoH, dbZ->special(), zShapeInfoD, dims.data(), dims.size());

    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execReduceSame2(Nd4jPointer *extraPointers,
                                int opNum,
                                OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                                void *extraParams,
                                OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
                                OpaqueDataBuffer *dbDimension, const Nd4jLong *hDimensionShape, const Nd4jLong *dDimensionShape) {
    try {
        auto dimension = reinterpret_cast<int *>(dbDimension->primary());
        int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

        std::vector<int> dimensions(dimension, dimension + dimensionLength);

        const auto zLen = shape::length(hZShapeInfo);

        const Nd4jLong* zShapeInfoH = hZShapeInfo;
        const Nd4jLong* zShapeInfoD = dZShapeInfo;

        if(shape::rank(hXShapeInfo) - dimensionLength != shape::rank(hZShapeInfo) && zLen != 1) {
            auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(hZShapeInfo, dimensions);
            zShapeInfoH = reinterpret_cast<Nd4jLong const*>(zPack.primary());
            zShapeInfoD = reinterpret_cast<Nd4jLong const*>(zPack.special());
        }

        std::vector<int> dims = (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(hXShapeInfo), dimensions) : std::vector<int>();
        NativeOpExecutioner::execReduceSame(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo, extraParams, dbZ->primary(), zShapeInfoH, dbZ->special(), zShapeInfoD, dims.data(), dims.size());

    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execReduceLong2(Nd4jPointer *extraPointers,
                                int opNum,
                                OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                                void *extraParams,
                                OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
                                OpaqueDataBuffer *dbDimension, const Nd4jLong *hDimensionShape, const Nd4jLong *dDimensionShape) {
    try {
        auto dimension = reinterpret_cast<int *>(dbDimension->primary());
        int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

        std::vector<int> dimensions(dimension, dimension + dimensionLength);

        const auto zLen = shape::length(hZShapeInfo);

        const Nd4jLong* zShapeInfoH = hZShapeInfo;
        const Nd4jLong* zShapeInfoD = dZShapeInfo;

        if(shape::rank(hXShapeInfo) - dimensionLength != shape::rank(hZShapeInfo) && zLen != 1) {
            auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(hZShapeInfo, dimensions);
            zShapeInfoH = reinterpret_cast<Nd4jLong const*>(zPack.primary());
            zShapeInfoD = reinterpret_cast<Nd4jLong const*>(zPack.special());
        }

        std::vector<int> dims = (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(hXShapeInfo), dimensions) : std::vector<int>();
        NativeOpExecutioner::execReduceLong(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo, extraParams, dbZ->primary(), zShapeInfoH, dbZ->special(), zShapeInfoD, dims.data(), dims.size());

    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParamsVals
 * @param hY
 * @param hYShapeInfo
 * @param hZ
 * @param hZShapeInfo
 */
void execReduce3(Nd4jPointer *extraPointers,
                                    int opNum,
                                    OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                                    void *extraParams,
                                    OpaqueDataBuffer *dbY, const Nd4jLong *hYShapeInfo, const Nd4jLong *dYShapeInfo,
                                    OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo) {
    try {
        NativeOpExecutioner::execReduce3(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo, extraParams, dbY->primary(), hYShapeInfo,
                                         dbY->special(), dYShapeInfo, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParamsVals
 * @param hY
 * @param hYShapeInfo
 */
void execReduce3Scalar(Nd4jPointer *extraPointers,int opNum,
                                            OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                                            void *extraParams,
                                            OpaqueDataBuffer *dbY, const Nd4jLong *hYShapeInfo, const Nd4jLong *dYShapeInfo,
                                            OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo) {
    try {
        NativeOpExecutioner::execReduce3Scalar(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo, extraParams, dbY->primary(),
                                               hYShapeInfo, dbY->special(), dYShapeInfo, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParamsVals
 * @param hY
 * @param hYShapeInfo
 * @param hZ
 * @param hZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void execReduce3Tad(Nd4jPointer *extraPointers,
                                    int opNum,
                                    OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                                    void *extraParams,
                                    OpaqueDataBuffer *dbY, const Nd4jLong *hYShapeInfo, const Nd4jLong *dYShapeInfo,
                                    OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
                                    OpaqueDataBuffer *dbDimension, const Nd4jLong *hDimensionShape, const Nd4jLong *dDimensionShape,
                                    const Nd4jLong *tadOnlyShapeInfo, const Nd4jLong *tadOffsets,
                                    const Nd4jLong *yTadOnlyShapeInfo, const Nd4jLong *yTadOffsets) {
    try {
        auto dimension = reinterpret_cast<int *>(dbDimension->primary());
        auto dimensionLength = static_cast<int>(shape::length(hDimensionShape));

        if (extraPointers == nullptr || extraPointers[2] == 0) {
            NativeOpExecutioner::execReduce3(LaunchContext::defaultContext(), opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                             extraParams, dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ->primary(), hZShapeInfo, dbZ->special(),
                                             dZShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets,
                                             yTadOnlyShapeInfo, yTadOffsets);
        } else {
            // going tad-way
            auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension,
                                                                                    dimensionLength);

            auto hTADShapeInfo = tadPack.primaryShapeInfo();
            auto hTADOffsets = tadPack.primaryOffsets();

            NativeOpExecutioner::execReduce3TAD(LaunchContext::defaultContext(), opNum, dbX->primary(), hXShapeInfo, dbX->special(),
                                                dXShapeInfo, extraParams, dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ->primary(),
                                                hZShapeInfo, dbZ->special(), dZShapeInfo, dimension, dimensionLength, hTADShapeInfo,
                                                hTADOffsets, nullptr, nullptr);
        }
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

bool isBlasVersionMatches(int major, int minor, int build) {
    return true;
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param hZ
 * @param hZShapeInfo
 * @param hScalar
 * @param extraParams
 * @param n
 */
void execScalar(
        Nd4jPointer *extraPointers,
        int opNum,
        OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
        OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
        OpaqueDataBuffer *dbScalar, const Nd4jLong *hScalarShapeInfo, const Nd4jLong *dScalarShapeInfo,
        void *extraParams) {
    try {
        NativeOpExecutioner::execScalar(nullptr,
                                        opNum,
                                        dbX->primary(),
                                        hXShapeInfo,
                                        dbX->special(),
                                        dXShapeInfo,
                                        dbZ->primary(),
                                        hZShapeInfo,
                                        dbZ->special(),
                                        dZShapeInfo,
                                        dbScalar->primary(),
                                        hScalarShapeInfo,
                                        dbScalar->special(),
                                        dScalarShapeInfo,
                                        extraParams);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execScalarBool(
        Nd4jPointer *extraPointers,
        int opNum,
        OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
        OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
        OpaqueDataBuffer *dbScalar, const Nd4jLong *hScalarShapeInfo, const Nd4jLong *dScalarShapeInfo,
        void *extraParams) {
    try {
        NativeOpExecutioner::execScalarBool(nullptr,
                                            opNum,
                                            dbX->primary(),
                                            hXShapeInfo,
                                            dbX->special(),
                                            dXShapeInfo,
                                            dbZ->primary(),
                                            hZShapeInfo,
                                            dbZ->special(),
                                            dZShapeInfo,
                                            dbScalar->primary(),
                                            hScalarShapeInfo,
                                            dbScalar->special(),
                                            dScalarShapeInfo,
                                            extraParams);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 */
void execSummaryStatsScalar(Nd4jPointer *extraPointers,
        int opNum,
        OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
        void *extraParams,
        OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
        bool biasCorrected) {
    try {
        NativeOpExecutioner::execSummaryStatsScalar(nullptr,
                                                    opNum,
                                                    dbX->primary(),
                                                    hXShapeInfo,
                                                    dbX->special(),
                                                    dXShapeInfo,
                                                    extraParams,
                                                    dbZ->primary(),
                                                    hZShapeInfo,
                                                    dbZ->special(),
                                                    dZShapeInfo,
                                                    biasCorrected);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @param hZ
 * @param hZShapeInfo
 */
void execSummaryStats(Nd4jPointer *extraPointers,
                                         int opNum,
                                         OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                                         void *extraParams,
                                         OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
                                         bool biasCorrected) {
    try {
        NativeOpExecutioner::execSummaryStats(nullptr,
                                              opNum,
                                              dbX->primary(),
                                              hXShapeInfo,
                                              dbX->special(),
                                              dXShapeInfo,
                                              extraParams,
                                              dbZ->primary(),
                                              hZShapeInfo,
                                              dbZ->special(),
                                              dZShapeInfo,
                                              biasCorrected);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @param hZ
 * @param hZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void execSummaryStatsTad(Nd4jPointer *extraPointers,
                                         int opNum,
                                         OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                                         void *extraParams,
                                         OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
                                         OpaqueDataBuffer *dbDimension, const Nd4jLong *hDimensionShape, const Nd4jLong *dDimensionShape,
                                         bool biasCorrected,
                         const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets) {
    try {
        auto dimension = reinterpret_cast<int *>(dbDimension->primary());
        int dimensionLength = static_cast<int>(shape::length(hDimensionShape));


        NativeOpExecutioner::execSummaryStats(nullptr,
                                              opNum,
                                              dbX->primary(),
                                              hXShapeInfo,
                                              dbX->special(),
                                              dXShapeInfo,
                                              extraParams,
                                              dbZ->primary(),
                                              hZShapeInfo,
                                              dbZ->special(),
                                              dZShapeInfo,
                                              dimension,
                                              dimensionLength,
                                              tadShapeInfo,
                                              tadOffsets,
                                              biasCorrected);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param hZ
 * @param hZShapeInfo
 * @param extraParams
 * @param n
 */
void execTransformFloat(
        Nd4jPointer *extraPointers,
        int opNum,
        OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
        OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
        void *extraParams) {
    try {
        NativeOpExecutioner::execTransformFloat(nullptr,
                                                opNum,
                                                dbX->primary(),
                                                hXShapeInfo,
                                                dbX->special(),
                                                dXShapeInfo,
                                                dbZ->primary(),
                                                hZShapeInfo,
                                                dbZ->special(),
                                                dZShapeInfo,
                                                extraParams,
                                                nullptr,
                                                nullptr);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execTransformSame(
        Nd4jPointer *extraPointers,
        int opNum,
        OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
        OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
        void *extraParams) {
    try {
        NativeOpExecutioner::execTransformSame(nullptr,
                                               opNum,
                                               dbX->primary(),
                                               hXShapeInfo,
                                               dbX->special(),
                                               dXShapeInfo,
                                               dbZ->primary(),
                                               hZShapeInfo,
                                               dbZ->special(),
                                               dZShapeInfo,
                                               extraParams,
                                               nullptr,
                                               nullptr);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execTransformBool(
        Nd4jPointer *extraPointers,
        int opNum,
        OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
        OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
        void *extraParams) {
    try {
        NativeOpExecutioner::execTransformBool(nullptr,
                                               opNum,
                                               dbX->primary(),
                                               hXShapeInfo,
                                               dbX->special(),
                                               dXShapeInfo,
                                               dbZ->primary(),
                                               hZShapeInfo,
                                               dbZ->special(),
                                               dZShapeInfo,
                                               extraParams,
                                               nullptr,
                                               nullptr);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execTransformAny(
        Nd4jPointer *extraPointers,
        int opNum,
        OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
        OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
        void *extraParams) {
    try {
        NativeOpExecutioner::execTransformAny(nullptr,
                                              opNum,
                                              dbX->primary(),
                                              hXShapeInfo,
                                              dbX->special(),
                                              dXShapeInfo,
                                              dbZ->primary(),
                                              hZShapeInfo,
                                              dbZ->special(),
                                              dZShapeInfo,
                                              extraParams,
                                              nullptr,
                                              nullptr);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execTransformStrict(
        Nd4jPointer *extraPointers,
        int opNum,
        OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
        OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
        void *extraParams) {
    try {
        NativeOpExecutioner::execTransformStrict(nullptr,
                                                 opNum,
                                                 dbX->primary(),
                                                 hXShapeInfo,
                                                 dbX->special(),
                                                 dXShapeInfo,
                                                 dbZ->primary(),
                                                 hZShapeInfo,
                                                 dbZ->special(),
                                                 dZShapeInfo,
                                                 extraParams,
                                                 nullptr,
                                                 nullptr);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execReduce3All(Nd4jPointer *extraPointers,
                                     int opNum,
                                     OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                                     void *extraParamsVals,
                                     OpaqueDataBuffer *dbY, const Nd4jLong *hYShapeInfo, const Nd4jLong *dYShapeInfo,
                                     OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
                                     OpaqueDataBuffer *dbDimension, const Nd4jLong *hDimensionShape, const Nd4jLong *dDimensionShape,
                                     const Nd4jLong *xTadShapeInfo, const Nd4jLong *xOffsets,
                                     const Nd4jLong *yTadShapeInfo, const Nd4jLong *yOffsets) {

    try {
        auto dimension = reinterpret_cast<int *>(dbDimension->primary());
        auto dimensionLength = static_cast<int>(shape::length(hDimensionShape));


        NativeOpExecutioner::execReduce3All(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo, extraParamsVals, dbY->primary(),
                                            hYShapeInfo, dbY->special(), dYShapeInfo, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo, dimension,
                                            dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

/**
    * Concatneate multi array of the same shape together
    * along a particular dimension
    */
void specialConcat(
        Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        void *hZ,
        Nd4jLong const* hZShapeInfo,
        Nd4jPointer *tadPointers,
        Nd4jPointer *offsetPointers) {
    try {
        auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

        BUILD_SINGLE_SELECTOR(zType, sd::SpecialMethods,::concatCpuGeneric(dimension, numArrays, data, inputShapeInfo, hZ, hZShapeInfo), LIBND4J_TYPES);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

/**
 * This is dummy method for JNI compatibility
 * Since we'll use this from java, jni compiler would like to have method no matter what.
 */
void initializeDevicesAndFunctions() {

}

void initializeFunctions(Nd4jPointer *functions) {
    sd::BlasHelper::getInstance().initializeFunctions(functions);
}

/**
       * This method acquires memory chunk of requested size on host side
       *
       * @param pointer pointer that'll be used for allocation
       * @param memorySize memory size, in bytes
       * @param flags optional parameter
       */
Nd4jPointer mallocHost(Nd4jLong memorySize, int flags) {
    return reinterpret_cast<Nd4jPointer>(new int8_t[memorySize]);
}

/**
 * This method acquires memory chunk of requested size on specified device
 *
 * PLEASE NOTE: This method is NOT supported and has NO effect in CPU-based backend.
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param ptrToDeviceId pointer to deviceId. For cuda that's just and int, for OpenCL that's pointer to device_id, etc
 * @param flags optional parameter
 */
Nd4jPointer mallocDevice(Nd4jLong memorySize, int deviceId, int flags) {
    // not supported
    return 0L;
}

/**
 * This method releases previously allocated host memory space
 *
 * @param pointer pointer that'll be freed
 */
int freeHost(Nd4jPointer pointer) {
    delete[] reinterpret_cast<int8_t *>(pointer);
    return 1L;
}

/**
 * This method releases previously allocated memory space on device
 *
 * PLEASE NOTE: This method is NOT supported and has NO effect in CPU-based backend.
 *
 * @param pointer pointer that'll be freed
 * @param ptrToDeviceId pointer to deviceId.
 */
int freeDevice(Nd4jPointer pointer, int deviceId) {
    // not supported
    return 0L;
}


/**
 * Returns the maximum number open mp threads
 */
int ompGetMaxThreads() {
    return omp_get_max_threads();
}

/**
 * Returns the number open mp threads
 */
int ompGetNumThreads() {
    return omp_get_num_threads();
}

/**
 * Sets the number of openmp threads
 */
void setOmpNumThreads(int threads) {
    omp_set_num_threads(threads);

}

Nd4jPointer createContext() {
    return 0L;
}

Nd4jPointer createStream() {
    return 0L;
}

Nd4jPointer createEvent() {
    return 0L;
}

int getDeviceMajor(int deviceId ) {
    return 0;
}

int getDeviceMinor(int deviceId) {
    return 0;
}

int registerEvent(Nd4jPointer event, Nd4jPointer stream) {
    return 0L;
}

int setDevice(int deviceId) {
    return 0L;
}

Nd4jLong getDeviceFreeMemory(int deviceId) {
    return 0L;
}

Nd4jLong getDeviceFreeMemoryDefault() {
    return 0L;
}

Nd4jLong getDeviceTotalMemory(int deviceId) {
    return 0L;
}

int memcpySync(Nd4jPointer dst, Nd4jPointer src, Nd4jLong size, int flags, Nd4jPointer reserved) {
    return 0L;
}

int memcpyAsync(Nd4jPointer dst, Nd4jPointer src, Nd4jLong size, int flags, Nd4jPointer reserved) {
    return 0L;
}

int memsetSync(Nd4jPointer dst, int value, Nd4jLong size, int flags, Nd4jPointer reserved) {
    return 0L;
}

int memsetAsync(Nd4jPointer dst, int value, Nd4jLong size,  int flags, Nd4jPointer reserved) {
    return 0L;
}

int destroyEvent(Nd4jPointer event) {
    return 0L;
}

int streamSynchronize(Nd4jPointer stream) {
    return 0L;
}

int eventSynchronize(Nd4jPointer event) {
    return 0L;
}

int getAvailableDevices() {
    return 0L;
}

void enableDebugMode(bool reallyEnable) {
    sd::Environment::getInstance().setDebug(reallyEnable);
}

void enableVerboseMode(bool reallyEnable) {
    sd::Environment::getInstance().setVerbose(reallyEnable);
}

void setGridLimit(int gridSize) {
    // no-op
}

sd::TadPack* tadOnlyShapeInfo(Nd4jLong const* hXShapeInfo, int *dimension, int dimensionLength) {
    auto pack = new TadPack();
    try {
        *pack = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }

    return pack;
}

Nd4jLong const* getPrimaryShapeInfo(sd::TadPack* pack) {
    return const_cast<Nd4jLong*>(pack->primaryShapeInfo());
}

Nd4jLong const* getPrimaryOffsets(sd::TadPack* pack) {
    return const_cast<Nd4jLong*>(pack->primaryOffsets());
}

Nd4jLong const* getSpecialShapeInfo(sd::TadPack* pack) {
    return const_cast<Nd4jLong*>(pack->specialShapeInfo());
}

Nd4jLong const* getSpecialOffsets(sd::TadPack* pack) {
    return const_cast<Nd4jLong*>(pack->specialOffsets());
}

Nd4jLong getNumberOfTads(sd::TadPack* pack) {
    return pack->numberOfTads();
}

int getShapeInfoLength(sd::TadPack* pack) {
    return pack->shapeInfoLength();
}

int memcpyConstantAsync(Nd4jLong dst, Nd4jPointer src, Nd4jLong size, int flags, Nd4jPointer reserved) {
    // no-op
    return 0L;
}

Nd4jPointer getConstantSpace() {
    // no-op
    return 0L;
}

template<typename T>
void pullRowsGeneric(void *vx,
                     Nd4jLong const* hXShapeInfo,
                     void *vz,
                     Nd4jLong const* hZShapeInfo,
                     const int n,
                     Nd4jLong const* indexes,
                     Nd4jLong const* tadShapeInfo,
                     Nd4jLong const* tadOffsets,
                     Nd4jLong const* zTadShapeInfo,
                     Nd4jLong const* zTadOffsets) {
    auto hX = reinterpret_cast<T *>(vx);
    auto hZ = reinterpret_cast<T *>(vz);

    const auto xEWS = shape::elementWiseStride(tadShapeInfo);
    const auto zEWS = shape::elementWiseStride(zTadShapeInfo);
    const auto tadLength = shape::length(tadShapeInfo);

    int elementsPerThread = n / TAD_THRESHOLD;
    int _threads = sd::math::nd4j_max<int>(1, elementsPerThread);
    _threads = sd::math::nd4j_min<int>(_threads, sd::Environment::getInstance().maxThreads());

    auto func = PRAGMA_THREADS_FOR {
        for (auto idx = start; idx < stop; idx++) {
            auto xTadOffsetForBlock = tadOffsets[indexes[idx]];
            auto zTadOffsetForBlock = zTadOffsets[idx];

            auto rX = hX + xTadOffsetForBlock;
            auto rZ = hZ + zTadOffsetForBlock;

            if (xEWS == 1 && zEWS == 1) {
                PRAGMA_OMP_SIMD
                for (Nd4jLong i = 0; i < tadLength; i++) {
                    rZ[i] = rX[i];
                }
            } else if (xEWS >= 1 && zEWS >= 1) {
                PRAGMA_OMP_SIMD
                for (Nd4jLong i = 0; i < tadLength; i++) {
                    rZ[i * zEWS] = rX[i * xEWS];
                }
            } else {
                for (Nd4jLong i = 0; i < tadLength; i++) {
                    auto xOffset = xTadOffsetForBlock + shape::getIndexOffset(i, tadShapeInfo);
                    auto zOffset = zTadOffsetForBlock + shape::getIndexOffset(i, zTadShapeInfo);
                    hZ[zOffset] = hX[xOffset];
                }
            }
        }
    };

    samediff::Threads::parallel_tad(func, 0, n, 1, _threads);
}

void pullRows(Nd4jPointer *extraPointers,
        OpaqueDataBuffer *dbX, Nd4jLong const* hXShapeInfo, Nd4jLong const* dXShapeInfo,
        OpaqueDataBuffer *dbZ, Nd4jLong const* hZShapeInfo, Nd4jLong const* dZShapeInfo,
        Nd4jLong n,
        Nd4jLong* indexes,
        Nd4jLong const* tadShapeInfo,
        Nd4jLong const* tadOffsets,
        Nd4jLong const* zTadShapeInfo,
        Nd4jLong const* zTadOffsets) {
    try {
        auto xType = sd::ArrayOptions::dataType(hXShapeInfo);

        BUILD_SINGLE_SELECTOR(xType, pullRowsGeneric, (dbX->primary(), hXShapeInfo, dbZ->primary(), hZShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets), LIBND4J_TYPES);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

template<typename T>
void tearGeneric(void *vx,
        Nd4jLong const* hXShapeInfo,
        Nd4jPointer *targets,
        Nd4jLong const* hZShapeInfo,
        Nd4jLong const* tadShapeInfo,
        Nd4jLong const* tadOffsets) {

    auto hX = reinterpret_cast<T *>(vx);

    const auto tadLength = shape::length(tadShapeInfo);
    auto tadEWS = shape::elementWiseStride(tadShapeInfo);
    auto zEWS = shape::elementWiseStride(hZShapeInfo);
    auto numTads = shape::length(hXShapeInfo) / tadLength;

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
            auto hZ = reinterpret_cast<T *>(targets[i]);
            auto s = hX + tadOffsets[i];

            if (zEWS == 1 && tadEWS == 1) {
                PRAGMA_OMP_SIMD
                for (Nd4jLong j = 0; j < tadLength; j++) {
                    hZ[j] = s[j];
                }
            } else if (zEWS > 0 && tadEWS > 0) {
                PRAGMA_OMP_SIMD
                for (Nd4jLong j = 0; j < tadLength; j++) {
                    hZ[j * zEWS] = s[j * tadEWS];
                }
            } else {
                for (Nd4jLong j = 0; j < tadLength; j++)
                    hZ[shape::getIndexOffset(j, hZShapeInfo)] = s[shape::getIndexOffset(j, tadShapeInfo)];
            }
        }
    };

    samediff::Threads::parallel_tad(func,0, numTads);
}

void tear(Nd4jPointer *extraPointers,
        OpaqueDataBuffer *dbX, Nd4jLong const* hXShapeInfo, Nd4jLong const* dXShapeInfo,
        Nd4jPointer *targets,
        Nd4jLong const* hZShapeInfo,
        Nd4jLong const* tadShapeInfo,
        Nd4jLong const* tadOffsets) {
    try {
        auto xType = sd::ArrayOptions::dataType(hXShapeInfo);

        BUILD_SINGLE_SELECTOR(xType, tearGeneric, (dbX->primary(), hXShapeInfo, targets, hZShapeInfo, tadShapeInfo, tadOffsets), LIBND4J_TYPES);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}


void average(Nd4jPointer *extras,
        Nd4jPointer *hX, const Nd4jLong *hXShapeInfo,
        Nd4jPointer *dX, const Nd4jLong *dXShapeInfo,
        void *z, const Nd4jLong *hZShapeInfo,
        void *dz, const Nd4jLong *dZShapeInfo,
        int n,
        Nd4jLong length,
        bool propagate) {
    try {
        auto xType = sd::ArrayOptions::dataType(hXShapeInfo);

        BUILD_SINGLE_SELECTOR(xType, sd::SpecialMethods, ::averageGeneric(hX, z, hZShapeInfo, n, length, propagate), LIBND4J_TYPES);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void accumulate(Nd4jPointer *extras,
        Nd4jPointer *hX, Nd4jLong const* hXShapeInfo,
        Nd4jPointer *dX, Nd4jLong const* dXShapeInfo,
        void *hz, Nd4jLong const* hZShapeInfo,
        void *dz, Nd4jLong const* dZShapeInfo,
        int n,
        Nd4jLong length) {
    try {
        auto xType = sd::ArrayOptions::dataType(hXShapeInfo);

        BUILD_SINGLE_SELECTOR(xType, sd::SpecialMethods, ::accumulateGeneric(hX, hz, hZShapeInfo, n, length), LIBND4J_TYPES);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void enableP2P(bool enable) {
    // no-op
}



void encodeThresholdP1(Nd4jPointer *extraPointers, void *hX, Nd4jLong const* hXShapeInfo, Nd4jLong N, int *dz, float threshold) {
    // TODO: to be implemented
}


void encodeThresholdP2Int(Nd4jPointer *extraPointers, int *hX, Nd4jLong N, int *dz) {
    // TODO: to be implemented
}


void encodeThresholdP3(Nd4jPointer *extraPointers, void *hX, Nd4jLong const* hXShapeInfo, int *offsets, Nd4jLong N, int *dz){
    // offsets won't be used here

    // TODO: to be implemented
}

void decodeThreshold(Nd4jPointer *extraPointers, void *hX, Nd4jLong N, void *dz, const Nd4jLong *hZShapeInfo){
    // TODO: to be implemented
}

bool isP2PAvailable() {
    // always TRUE for cpu backend
    return true;
}

void checkP2P() {
    // no-op
}

void decodeBitmap(Nd4jPointer *extraPointers, void *hX, Nd4jLong N, void *dz, Nd4jLong  const* hZShapeInfo) {
    NativeOpExecutioner::decodeBitmap(hX, N, dz, hZShapeInfo);
}

template<typename T>
void shuffleGeneric(void **hX, Nd4jLong * const*hXShapeInfo, void **dz, Nd4jLong * const* hZShapeInfo, int N, int *shuffleMap, Nd4jLong * const* tadOnlyShapeInfo, Nd4jLong * const* tadOffsets) {

    auto dX = reinterpret_cast<T **>(hX);
    auto dZ = reinterpret_cast<T **>(dz);

    auto func = PRAGMA_THREADS_FOR {
        for (auto f = start; f < stop; f++) {
            auto hX = reinterpret_cast<T *>(dX[f]);
            //auto hZ = reinterpret_cast<T *>(dZ[f]);

            auto xShapeInfo = hXShapeInfo[f];
            auto tadOffset = reinterpret_cast<Nd4jLong *>(tadOffsets[f]);


            const auto tadLength = shape::length(tadOnlyShapeInfo[f]);
            auto tadEWS = shape::elementWiseStride(tadOnlyShapeInfo[f]);
            auto tadRank = shape::rank(tadOnlyShapeInfo[f]);
            auto numTads = shape::length(hXShapeInfo[f]) / tadLength;

            auto tadShape = shape::shapeOf(tadOnlyShapeInfo[f]);
            auto tadStride = shape::stride(tadOnlyShapeInfo[f]);

            if (shape::rank(xShapeInfo) == 1) {
                auto xLength = shape::length(xShapeInfo);
                auto ews = shape::elementWiseStride(xShapeInfo);
                for (Nd4jLong r = 0; r < xLength; r++) {
                    auto swapIdx = shuffleMap[r];
                    if (swapIdx < 0)
                        continue;

                    sd::math::nd4j_swap<T>(hX[r * ews], hX[swapIdx * ews]);
                }
            } else {
                for (Nd4jLong r = 0; r < numTads; r++) {
                    if (shuffleMap[r] < 0)
                        continue;

                    auto oldOffset = tadOffset[r];
                    auto newOffset = tadOffset[shuffleMap[r]];

                    auto rX = hX + oldOffset;
                    auto rY = hX + newOffset;

                    if (tadEWS == 1) {
                        for (Nd4jLong i = 0; i < tadLength; i++) {
                            sd::math::nd4j_swap<T>(rX[i], rY[i]);
                        }
                    } else {
                        for (Nd4jLong i = 0; i < tadLength; i++) {
                            auto offset = shape::getIndexOffset(i, tadOnlyShapeInfo[f]);
                            sd::math::nd4j_swap<T>(hX[offset + oldOffset], hX[offset + newOffset]);
                        }
                    }
                }
            }
        }
    };

    samediff::Threads::parallel_tad(func, 0, N);
}

void shuffle(Nd4jPointer *extras,
                              Nd4jPointer *hX, Nd4jPointer *hXShapeInfo,
                              Nd4jPointer *dX, Nd4jPointer *dXShapeInfo,
                              Nd4jPointer *hz, Nd4jPointer *hZShapeInfo,
                              Nd4jPointer *dz, Nd4jPointer *dZShapeInfo,
                              int N,
                              int *shuffleMap,
                              Nd4jPointer *tadShapeInfo,
                              Nd4jPointer *tadOffsets) {
    try {
        auto xShape = reinterpret_cast<Nd4jLong * const*>(hXShapeInfo);
        auto zShape = reinterpret_cast<Nd4jLong * const*>(hZShapeInfo);
        auto tadOnlyShapeInfo = reinterpret_cast<Nd4jLong * const*>(tadShapeInfo);
        auto tadOffset = reinterpret_cast<Nd4jLong * const*>(tadOffsets);

        auto xType = sd::ArrayOptions::dataType(xShape[0]);

        BUILD_SINGLE_SELECTOR(xType, shuffleGeneric,
                              (hX, xShape, hz, zShape, N, shuffleMap, tadOnlyShapeInfo, tadOffset), LIBND4J_TYPES);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}


bool isExperimentalEnabled() {
    return sd::Environment::getInstance().isExperimentalBuild();
}


void setOmpMinThreads(int threads) {
    // TODO: to be implemented
}

int getDevice() {
    return 0;
}

void execScalarTad(Nd4jPointer *extraPointers,
                                 int opNum,
                                 OpaqueDataBuffer *dbX, Nd4jLong  const* hXShapeInfo, Nd4jLong  const*dXShapeInfo,
                                 OpaqueDataBuffer *dbZ, Nd4jLong  const* hZShapeInfo, Nd4jLong  const*dZShapeInfo,
                                 OpaqueDataBuffer *dbScalars, Nd4jLong const* hScalarShapeInfo, Nd4jLong  const* dScalarShapeInfo,
                                 void *extraParams,
                                 OpaqueDataBuffer *dbDimension, Nd4jLong const* hDimensionShape, Nd4jLong const* dDimensionShape,
                                 Nd4jLong const*tadShapeInfo, Nd4jLong const* tadOffsets,
                                 Nd4jLong const*tadShapeInfoZ, Nd4jLong const* tadOffsetsZ) {
    try {
        auto dimension = reinterpret_cast<int *>(dbDimension->primary());
        int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

        NativeOpExecutioner::execScalar(nullptr,
                                        opNum,
                                        dbX->primary(),
                                        hXShapeInfo,
                                        dbX->special(),
                                        dXShapeInfo,
                                        extraParams,
                                        dbZ->primary(),
                                        hZShapeInfo,
                                        dbZ->special(),
                                        dZShapeInfo,
                                        dbScalars->primary(),
                                        hScalarShapeInfo,
                                        dbScalars->special(),
                                        dScalarShapeInfo,
                                        dimension,
                                        shape::length(hDimensionShape),
                                        tadShapeInfo,
                                        tadOffsets,
                                        tadShapeInfoZ,
                                        tadOffsetsZ);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execScalarBoolTad(Nd4jPointer *extraPointers,
                           int opNum,
                           OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                           OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
                           OpaqueDataBuffer *dbScalars, const Nd4jLong *hScalarShapeInfo, const Nd4jLong *dScalarShapeInfo,
                           void *extraParams,
                           OpaqueDataBuffer *dbDimension, const Nd4jLong *hDimensionShape, const Nd4jLong *dDimensionShape,
                           const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets,
                           const Nd4jLong *tadShapeInfoZ, const Nd4jLong *tadOffsetsZ) {
    try {
        auto dimension = reinterpret_cast<int *>(dbDimension->primary());
        int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

        NativeOpExecutioner::execScalarBool(nullptr,
                                            opNum,
                                            dbX->primary(),
                                            hXShapeInfo,
                                            dbX->special(),
                                            dXShapeInfo,
                                            extraParams,
                                            dbZ->primary(),
                                            hZShapeInfo,
                                            dbZ->special(),
                                            dZShapeInfo,
                                            dbScalars->primary(),
                                            hScalarShapeInfo,
                                            dbScalars->special(),
                                            dScalarShapeInfo,
                                            dimension,
                                            dimensionLength,
                                            tadShapeInfo,
                                            tadOffsets,
                                            tadShapeInfoZ,
                                            tadOffsetsZ);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

const char * getDeviceName(int deviceId) {
    try {
        if (!nameSet) {
            name = reinterpret_cast<char *>(malloc(256 * sizeof(char)));

            CHECK_ALLOC(name, "Failed to allocate new string buffer", 256);

            std::memset(name, 0, 256 * sizeof(char));
            nameSet = true;

            // TODO: provide proper CPU model name here
            sprintf(name, "x86-compatible CPU");
        }
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }


    return name;
}


void execAggregate(Nd4jPointer *extraPointers,int opNum,
                                    void **arguments,
                                    int numArguments,
                                    Nd4jLong **shapeArguments,
                                    int numShapeArguments,
                                    int *indexArguments,
                                    int numIndexArguments,
                                    int **intArrays,
                                    int numIntArrays,
                                    void *realArguments,
                                    int numRealArguments,
                                    sd::DataType dtype) {

}

void batchExecutor(Nd4jPointer *extraPointers,
                               int numAggregates,
                               int opNum,
                               int maxArgs,
                               int maxShapes,
                               int maxIntArrays,
                               int maxIntArraySize,
                               int maxIdx,
                               int maxReals,
                               void *ptrToArguments,
                               sd::DataType dtype) {

}

void execAggregateBatch(Nd4jPointer *extraPointers,
                                         int numAggregates,
                                         int opNum,
                                         int maxArgs,
                                         int maxShapes,
                                         int maxIntArrays,
                                         int maxIntArraySize,
                                         int maxIdx,
                                         int maxReals,
                                         void *ptrToArguments,
                                         sd::DataType dtype) {

}


void execRandom(Nd4jPointer *extraPointers,
                                 int opNum,
                                 Nd4jPointer state,
                                 OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
                                 void *extraArguments) {
    try {
        NativeOpExecutioner::execRandom(nullptr, opNum, state, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo, extraArguments);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execRandom3(Nd4jPointer *extraPointers,
                                 int opNum,
                                 Nd4jPointer state,
                                 OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                                 OpaqueDataBuffer *dbY, const Nd4jLong *hYShapeInfo, const Nd4jLong *dYShapeInfo,
                                 OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
                                 void *extraArguments) {
    try {
        NativeOpExecutioner::execRandom(nullptr, opNum, state, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo, dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo, extraArguments);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void execRandom2(Nd4jPointer *extraPointers,
                                 int opNum,
                                 Nd4jPointer state,
                                 OpaqueDataBuffer *dbX, const Nd4jLong *hXShapeInfo, const Nd4jLong *dXShapeInfo,
                                 OpaqueDataBuffer *dbZ, const Nd4jLong *hZShapeInfo, const Nd4jLong *dZShapeInfo,
                                 void *extraArguments) {
    try {
        NativeOpExecutioner::execRandom(nullptr, opNum, state, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo, extraArguments);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

Nd4jPointer initRandom(Nd4jPointer *extraPointers, long seed, long bufferSize, Nd4jPointer ptrToBuffer) {
    try {
        auto generator = new graph::RandomGenerator(seed, seed);

        return (Nd4jPointer) generator;
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());

        return nullptr;
    }
}

void refreshBuffer(Nd4jPointer *extraPointers, long seed, Nd4jPointer ptrRandom) {
    auto generator = reinterpret_cast<sd::graph::RandomGenerator*> (ptrRandom);

    generator->setStates(seed);
}

void reSeedBuffer(Nd4jPointer *extraPointers, long seed, Nd4jPointer ptrRandom) {
    auto generator = reinterpret_cast<sd::graph::RandomGenerator *> (ptrRandom);

    generator->setStates(seed);
}


void destroyRandom(Nd4jPointer ptrBuffer) {
    auto buffer = reinterpret_cast<sd::graph::RandomGenerator*>(ptrBuffer);
    delete buffer;
}




/**
    * Return the length of a shape buffer
    * based on the pointer
    * @param buffer  the buffer pointer to check
    * @return
    */
int lengthForShapeBufferPointer(Nd4jPointer buffer) {
    auto shapeBuffer = reinterpret_cast<Nd4jLong *>(buffer);
    return shape::shapeInfoLength(shape::rank(shapeBuffer));
}


/**
  * The pointer to get the address for
  *
  * @param address the address to get the pointer
  * @return the pointer for the given address
  */

Nd4jPointer pointerForAddress(Nd4jLong address) {
    return reinterpret_cast<Nd4jPointer >(address);
}

void sort(Nd4jPointer *extraPointers,
        void *hX, const Nd4jLong *hXShapeInfo,
        void *dX, const Nd4jLong *dXShapeInfo,
        bool descending) {
    try {
        NativeOpExecutioner::execSort(hX, hXShapeInfo, descending);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void sortTad(Nd4jPointer *extraPointers,
            void *hX, const Nd4jLong *hXShapeInfo,
            void *dX, const Nd4jLong *dXShapeInfo,
            int *dimension, int dimensionLength,
            const Nd4jLong *tadShapeInfo,
            const Nd4jLong *tadOffsets,
            bool descending) {
    try {
        NativeOpExecutioner::execSort(hX, hXShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void sortCooIndices(Nd4jPointer *extraPointers,
        Nd4jLong *indices,
        void *values,
        Nd4jLong length,
        int rank) {
    try {
        NativeOpExecutioner::execSortCooIndices(indices, values, length, rank);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

Nd4jLong encodeBitmap(Nd4jPointer *extraPointers, void *hX, Nd4jLong const* hXShapeInfo, Nd4jLong N, int *dz, float threshold) {
    return NativeOpExecutioner::encodeBitmap(hX, hXShapeInfo, N, dz, threshold);
}



Nd4jLong* mmapFile(Nd4jPointer *extraPointers, const char *fileName, Nd4jLong length) {
    auto hZ = new Nd4jLong[2];errno = 0;
try {
#if defined(_WIN32) || defined(_WIN64)
    _mmap(hZ, static_cast<size_t>(length), fileName);
#else
    int fd = open(fileName, O_RDWR, 0);// checking for failed fopen
    if (fd < 0) {
        nd4j_printf("Errno: %i\n", errno);
        throw std::runtime_error("Failed to open file for MMAP");
    }
    void *ptr = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

// check for failed allocation
    if (ptr == MAP_FAILED)
        return nullptr;

    hZ[0] = (Nd4jLong) ptr;
    hZ[1] = fd;

#endif

    return hZ;
} catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
}
}

void munmapFile(Nd4jPointer *extraPointers, Nd4jLong *ptrMap, Nd4jLong length) {
    munmap((Nd4jPointer) ptrMap[0], length);
#if defined(_WIN32) || defined(_WIN64)
    CloseHandle(reinterpret_cast<HANDLE>(ptrMap[1]));
#else
    close((int) ptrMap[1]);
#endif

    delete[] ptrMap;
}

sd::graph::ResultWrapper* executeFlatGraph(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer) {
    try {
        return sd::graph::GraphExecutioner::executeFlatBuffer(flatBufferPointer);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return nullptr;
    }
}

Nd4jLong getResultWrapperSize(sd::graph::ResultWrapper* ptr) {
    return ptr->size();
}
Nd4jPointer getResultWrapperPointer(sd::graph::ResultWrapper* ptr) {
    return ptr->pointer();
}

const char* getAllCustomOps() {
    return sd::ops::OpRegistrator::getInstance().getAllCustomOperations();
}

template <typename T>
FORCEINLINE int estimateThresholdGeneric(Nd4jPointer *extraPointers, Nd4jPointer hX, int N, T threshold) {
    auto buffer = reinterpret_cast<T *>(hX);
    int span = (N / 6) + 8;

    auto func = PRAGMA_REDUCE_LONG {
        int64_t cnt = 0;
        PRAGMA_OMP_SIMD
        for (auto e = start; e < stop; e++) {
            auto v = sd::math::nd4j_abs<T>(buffer[e]);
            if (v >= threshold)
                cnt++;
        }

        return cnt;
    };

    return samediff::Threads::parallel_long(func, LAMBDA_AL { return _old + _new; }, 0, N);
}


int estimateThreshold(Nd4jPointer *extraPointers, Nd4jPointer hX, Nd4jLong const* hXShapeInfo, int N, float threshold) {
    try {
        auto xType = ArrayOptions::dataType(hXShapeInfo);
        BUILD_SINGLE_SELECTOR(xType, return estimateThresholdGeneric, (extraPointers, hX, N, threshold), FLOAT_TYPES);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return 0;
    }
}

Nd4jLong getShapeListSize(sd::ShapeList* list) {
    return list->size();
}

Nd4jLong const* getShape(sd::ShapeList* list, Nd4jLong i) {
    return const_cast<Nd4jLong const*>(list->at(i));
}

void deleteShapeList(Nd4jPointer shapeList) {
    auto list = reinterpret_cast<sd::ShapeList*>(shapeList);

    //list->destroy();
    delete list;
}

sd::ShapeList* _calculateOutputShapes(Nd4jPointer* extraPointers, sd::ops::DeclarableOp* op, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool *bArgs, int numBArgs, int *dArgs, int numDArgs) {
    sd::graph::VariableSpace varSpace;
    Context block(2, &varSpace);
    sd::ShapeList inShapes;

    for (int e = 0; e < numIArgs; e++)
        block.getIArguments()->push_back(iArgs[e]);

    for (int e = 0; e < numTArgs; e++)
        block.getTArguments()->push_back(tArgs[e]);

    for (int e = 0; e < numBArgs; e++)
        block.getBArguments()->push_back(bArgs[e]);

    for (int e = 0; e < numDArgs; e++)
        block.getDArguments()->push_back((sd::DataType) dArgs[e]);

    for (int e = 0; e < numInputShapes; e++) {
        auto shape_ = reinterpret_cast<Nd4jLong *>(inputShapes[e]);

        // we shouldn't copy buffer if that's empty array
        void *buffer_ = sd::ArrayOptions::arrayType(shape_) == ArrayType::EMPTY ? nullptr : inputBuffers[e];

        auto array = new sd::NDArray(buffer_, shape_, varSpace.launchContext(), false);

        // block should contain references to proper variable
        varSpace.putVariable(1, e, array);
        block.pickInput(1, e);

        inShapes.push_back(shape_);
    }

    auto status = op->validateDataTypes(block);
    if (status != Status::OK())
        throw std::runtime_error("Data types validation failed");

    auto shapeList = op->calculateOutputShape(&inShapes, block);

    if (varSpace.launchContext() != nullptr)
        shapeList->detach();

    return shapeList;
}

sd::ShapeList* calculateOutputShapes2(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool *bArgs, int numBArgs, int *dArgs, int numDArgs) {
    try {
        auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);

        return _calculateOutputShapes(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs, bArgs, numBArgs, dArgs, numDArgs);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return nullptr;
    }
}

sd::ShapeList* _calculateOutputShapes(Nd4jPointer* extraPointers, sd::ops::DeclarableOp *op, Nd4jPointer* inputShapes, int numInputShapes, double *tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    Context block(1);
    sd::ShapeList inShapes;

    for (int e = 0; e < numIArgs; e++)
        block.getIArguments()->push_back(iArgs[e]);

    for (int e = 0; e < numTArgs; e++)
        block.getTArguments()->push_back(tArgs[e]);

    for (int e = 0; e < numInputShapes; e++)
        inShapes.push_back(reinterpret_cast<Nd4jLong *>(inputShapes[e]));

    auto shapeList = op->calculateOutputShape(&inShapes, block);
    shapeList->detach();

    return shapeList;
}

sd::ShapeList* calculateOutputShapes(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    try {
        auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);

        return _calculateOutputShapes(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return nullptr;
    }
}

int execCustomOp2(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer opContext) {
    try {
        auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);
        auto context = reinterpret_cast<Context *>(opContext);

        return op->execute(context);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return 20;
    }
}

Nd4jStatus realExec(sd::ops::DeclarableOp* op, Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool* bArgs, int numBArgs, bool isInplace) {
    if (op == nullptr)
        nd4j_printf("Can't find requested operation: [%lld]\n", hash);

    // we're using the same fake nodeId everywhere here

    std::vector<sd::NDArray*> inputs(numInputs);
    std::vector<sd::NDArray*> outputs(numOutputs);
    std::vector<double> ttArgs(numTArgs);
    std::vector<Nd4jLong> iiArgs(numIArgs);
    std::vector<bool> biArgs(numBArgs);

    // filling block now with inputs
    for (int e = 0; e < numInputs; e++) {
        auto shape = reinterpret_cast<Nd4jLong *>(inputShapes[e]);
        void *buffer = sd::ArrayOptions::arrayType(shape) == ArrayType::EMPTY ? nullptr : inputBuffers[e];

        inputs[e] = new sd::NDArray(buffer, shape);
    }

    // if not inplace - transferring output arrays

    if (!isInplace)
        for (int e = 0; e < numOutputs; e++) {
            // we want to keep original output shape intact
            auto shape = shape::copyShape(reinterpret_cast<Nd4jLong *>(outputShapes[e]));
            void *buffer = sd::ArrayOptions::arrayType(shape) == ArrayType::EMPTY ? nullptr : outputBuffers[e];

            // FIXME: revisit this.
            bool canNullify = true;
            for (int i = 0; i < numInputs; i++) {
                void *ibuffer = sd::ArrayOptions::arrayType(shape) == ArrayType::EMPTY ? nullptr : inputBuffers[i];
                if (ibuffer == buffer) {
                    canNullify = false;
                    break;
                }
            }

            if (canNullify)
                memset((uint8_t *) buffer, '\0', shape::length(shape) * DataTypeUtils::sizeOfElement(ArrayOptions::dataType(shape)));

            auto array = new sd::NDArray(buffer, shape);
            outputs[e] = array;

            // and we want to release shape copy once we're done
            delete []shape;
        }

    for (int e = 0; e < numIArgs; e++)
        iiArgs[e] = iArgs[e];


    for (int e = 0; e < numTArgs; e++)
        ttArgs[e] = tArgs[e];

    for (int e = 0; e < numBArgs; e++)
        biArgs[e] = bArgs[e];

    // hypothetically at this point we have everything filled
    auto hZ = op->execute(inputs, outputs, ttArgs, iiArgs, biArgs, std::vector<sd::DataType>(), isInplace);
    //auto hZ = op->execute(inputs, ttArgs, iiArgs, isInplace);



    if (!isInplace)
        for (int e = 0; e < numOutputs; e++) {
            //shape::printShapeInfoLinear("JVM output shape", (int *) outputShapes[e]);
            //shape::printShapeInfoLinear("C++ output shape", (int *) outputs[e]->shapeInfo());
            //outputs[e]->printIndexedBuffer("C++ raw output");
            //outputs[e]->printBuffer("C++ indexed output");

            if (outputs[e]->ordering() != shape::order(reinterpret_cast<Nd4jLong *>(outputShapes[e])))
                outputs[e]->streamline(shape::order(reinterpret_cast<Nd4jLong *>(outputShapes[e])));
        }

    for (auto v: inputs)
        delete v;

    for (auto v: outputs)
        delete v;

    return hZ;
}


int execCustomOp(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool* bArgs, int numBArgs, bool isInplace) {
    try {
        auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);
        return realExec(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs, tArgs, numTArgs, iArgs, numIArgs, bArgs, numBArgs, isInplace);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return 1;
    }
}

int registerGraph(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer flatBufferPointer) {
    try {
        auto graph = sd::graph::GraphExecutioner::importFromFlatPointer(flatBufferPointer);

        sd::graph::GraphHolder::getInstance().registerGraph(graphId, graph);

        return ND4J_STATUS_OK;
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return 1;
    }
}

static VariablesSet* executeStoredGraphT(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
    auto graph = sd::graph::GraphHolder::getInstance().cloneGraph(graphId);
    auto varSpace = graph->getVariableSpace();

    std::vector<sd::NDArray*> handles;

    for (int e = 0; e < numInputs; e++) {
        auto idx = inputIndices[e];

        // we'll delete this array later, together with cloned VariableSpace
        auto array = new sd::NDArray(inputBuffers[e], reinterpret_cast<Nd4jLong *>(inputShapes[e]));
        handles.emplace_back(array);

        if (varSpace->hasVariable(idx)) {
            auto var = varSpace->getVariable(idx);
            if (var->hasNDArray())
                delete var->getNDArray();

            var->setNDArray(array);
        } else
            varSpace->putVariable(idx, array);
    }

    auto hZ = sd::graph::GraphExecutioner::execute(graph, varSpace);
    auto varSet = new sd::graph::VariablesSet(hZ);

    if (hZ == ND4J_STATUS_OK) {
        // pull back results, and provide them
        auto outputs = graph->fetchOutputs();
        for (int e = 0; e < outputs->size(); e++) {
            // we're only getting variable ID/Index from original grap. values will be taken from cloned workspace
            std::pair<int, int> varId(outputs->at(e)->id(), outputs->at(e)->index());

            auto var = varSpace->getVariable(varId);

            varSet->push_back(var->clone());
        }

        delete outputs;
    }

    delete graph;

    return varSet;
}

sd::graph::VariablesSet* executeStoredGraph(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
    return nullptr;
}

Nd4jLong getVariablesSetSize(sd::graph::VariablesSet* set) {
    return set->size();
}

Nd4jStatus getVariablesSetStatus(sd::graph::VariablesSet* set) {
    return set->status();
}

sd::graph::Variable* getVariable(sd::graph::VariablesSet* set, Nd4jLong i) {
    return set->at(i);
}

int getVariableId(sd::graph::Variable* variable) {
    return variable->id();
}

int getVariableIndex(sd::graph::Variable* variable) {
    return variable->index();
}

const char* getVariableName(sd::graph::Variable* variable) {
    return variable->getName()->c_str();
}

Nd4jLong const* getVariableShape(sd::graph::Variable* variable) {
    return const_cast<Nd4jLong const*>(variable->getNDArray()->shapeInfo());
}

void* getVariableBuffer(sd::graph::Variable* variable) {
    return variable->getNDArray()->buffer();
}

int unregisterGraph(Nd4jPointer *extraPointers, Nd4jLong graphId) {

    sd::graph::GraphHolder::getInstance().dropGraphAny(graphId);

    return sd::Status::OK();
}

void deletePointerArray(Nd4jPointer pointer) {
    auto ptr = reinterpret_cast<Nd4jPointer *>(pointer);
    delete[] ptr;
}

void deleteCharArray(Nd4jPointer pointer) {
    auto ptr = reinterpret_cast<char *>(pointer);
    delete[] ptr;
}

void deleteIntArray(Nd4jPointer pointer) {
    auto ptr = reinterpret_cast<int *>(pointer);
    delete[] ptr;
}

void deleteLongArray(Nd4jPointer pointer) {
    auto ptr = reinterpret_cast<Nd4jLong *>(pointer);
    delete[] ptr;
}

void deleteVariablesSet(sd::graph::VariablesSet* pointer) {
    delete pointer;
}

const char* getAllOperations() {
    return sd::OpTracker::getInstance().exportOperations();
}


Nd4jPointer getGraphState(Nd4jLong id) {
    return (Nd4jPointer) new sd::graph::GraphState(id);
}

void deleteGraphState(Nd4jPointer state) {
    auto stateP = reinterpret_cast<sd::graph::GraphState*>(state);
    delete stateP;
}

Nd4jStatus execCustomOpWithScope_(Nd4jPointer *extraPointers, sd::graph::GraphState *state, Nd4jLong opHash, Nd4jLong *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs) {
    /**
     * That's basically exec, with VariableSpace provided in GraphState:
     * depending on operation (i.e. while of if), different logic executors could be used
     */

    auto graph = state->graph();
    auto varSpace = state->variableSpace();

    // Node is dynamically created, and has nothing beyond it: only inputs and outputs
    // this node has id of 0, and inputs are
    Node node(OpType_LOGIC, opHash, 0);

    // mapping inputs
    for (int e = 0; e < numInputs; e++) {
        auto buffer = inputBuffers[e];
        auto shapeInfo = reinterpret_cast<Nd4jLong *>(inputShapes[e]);

        auto array = new sd::NDArray(buffer, shapeInfo, varSpace->launchContext());

        // now we just put array to VarSpace
        varSpace->putVariable(0, e, array);
        node.pickInput(0, e);
    }

    // mapping scopes
    for (int e = 0; e < numScopes; e++) {
        // we should check scope existence in GraphState/Graph
        int scopeId = (int) scopes[e];
        if (!state->hasScope(scopeId)) {
            // nd4j_printf("execCustomOpWithScope: referenced scope [%i] doesn't exist\n", scopeId);
            return Status::THROW();
        }
        node.pickInput(scopeId, 0);
    }

    auto hZ = LogicExecutor::processNode(graph, &node);
    if (hZ != Status::OK())
        return hZ;

    // mapping outputs

    for (int e = 0; e < numOutputs; e++) {
        auto buffer = outputBuffers[e];
        auto shapeInfo = reinterpret_cast<Nd4jLong *>(outputShapes[e]);

        NDArray array(buffer, shapeInfo, varSpace->launchContext());

        // now we just put array to VarSpace to the same ID
        //varSpace->putVariable(0, e, array);

        auto t = varSpace->getVariable(0, e)->getNDArray();
        array.assign(t);
    }

    // removing input variables
    for (int e = 0; e < numInputs; e++) {
        varSpace->dropVariable(0, e);
    }


    // after some bla-bla-bla we should have Graph and Node for current op
    return Status::OK();
}

Nd4jStatus execCustomOpWithScope(Nd4jPointer *extraPointers, Nd4jPointer state, Nd4jLong opHash, Nd4jLong *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs) {
    try {
        return execCustomOpWithScope_(extraPointers, reinterpret_cast<sd::graph::GraphState *>(state), opHash, scopes, numScopes, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return 1;
    }
}

void deleteResultWrapper(Nd4jPointer ptr) {
    // just 0 room for compiler s@!t
    auto p = reinterpret_cast<sd::graph::ResultWrapper *>(ptr);
    delete p;
}

/*
 * TypeDef:
 *     void convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer hX, long N, int dstType, Nd4jPointer hZ);
 */
void convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer hX, Nd4jLong N, int dstType, Nd4jPointer hZ) {
    auto hx = reinterpret_cast<void *>(hX);
    auto hz = reinterpret_cast<void *>(hZ);

    if (srcType == ND4J_FLOAT8) {
        if (dstType == ND4J_FLOAT8) {
            // convertGeneric<double, sd::float8>(hx, N, hz);
        } else if (dstType == ND4J_INT8) {
            //sd::TypeCast::convertGeneric<sd::float8, sd::int8>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_UINT8) {
            //sd::TypeCast::convertGeneric<sd::float8, sd::uint8>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_FLOAT16) {
            //sd::TypeCast::convertGeneric<sd::float8, float16>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_INT16) {
            //sd::TypeCast::convertGeneric<sd::float8, sd::int16>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_UINT16) {
            //sd::TypeCast::convertGeneric<sd::float8, sd::uint16>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_FLOAT32) {
            //sd::TypeCast::convertGeneric<sd::float8, float>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_DOUBLE) {
            //sd::TypeCast::convertGeneric<sd::float8, double>(nullptr, hx, N, hz);
        } else {
            //nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_INT8) {
        if (dstType == ND4J_FLOAT8) {
            //sd::TypeCast::convertGeneric<sd::int8, sd::float8>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_INT8) {
            //convertGeneric<sd::int8, sd::int8>(hx, N, hz);
        } else if (dstType == ND4J_UINT8) {
            sd::TypeCast::convertGeneric<int8_t, uint8_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_FLOAT16) {
            sd::TypeCast::convertGeneric<int8_t, float16>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_INT16) {
            sd::TypeCast::convertGeneric<int8_t, int16_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_UINT16) {
            //sd::TypeCast::convertGeneric<int8_t, uint16_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO: eventually we might want to add it
        } else if (dstType == ND4J_FLOAT32) {
            sd::TypeCast::convertGeneric<int8_t, float>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_DOUBLE) {
            sd::TypeCast::convertGeneric<int8_t, double>(nullptr, hx, N, hz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_UINT8) {
        if (dstType == ND4J_FLOAT8) {
        //    sd::TypeCast::convertGeneric<uint8_t, sd::float8>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_INT8) {
            sd::TypeCast::convertGeneric<uint8_t, int8_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_UINT8) {
            sd::TypeCast::convertGeneric<uint8_t, uint8_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_FLOAT16) {
            sd::TypeCast::convertGeneric<uint8_t, float16>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_INT16) {
            sd::TypeCast::convertGeneric<uint8_t, int16_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_UINT16) {
     //       sd::TypeCast::convertGeneric<uint8_t, uint16_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO: still might want to add
        } else if (dstType == ND4J_FLOAT32) {
            sd::TypeCast::convertGeneric<uint8_t, float>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_DOUBLE) {
            sd::TypeCast::convertGeneric<uint8_t, double>(nullptr, hx, N, hz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_FLOAT16) {
        if (dstType == ND4J_FLOAT8) {
        //    sd::TypeCast::convertGeneric<float16, sd::float8>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_INT8) {
            sd::TypeCast::convertGeneric<float16, int8_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_UINT8) {
            sd::TypeCast::convertGeneric<float16, uint8_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_FLOAT16) {
            sd::TypeCast::convertGeneric<float16, float16>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_INT16) {
            sd::TypeCast::convertGeneric<float16, int16_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_UINT16) {
//            sd::TypeCast::convertGeneric<float16, uint16_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO: .... ^^^
        } else if (dstType == ND4J_FLOAT32) {
            sd::TypeCast::convertGeneric<float16, float>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_DOUBLE) {
            sd::TypeCast::convertGeneric<float16, double>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_THRESHOLD) {
            sd::TypeCast::convertToThreshold<float16>(nullptr, hx, N, hz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_INT16) {
        if (dstType == ND4J_FLOAT8) {
         //   sd::TypeCast::convertGeneric<int16_t, sd::float8>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_INT8) {
            sd::TypeCast::convertGeneric<int16_t, int8_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_UINT8) {
            sd::TypeCast::convertGeneric<int16_t, uint8_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_FLOAT16) {
            sd::TypeCast::convertGeneric<int16_t, float16>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_INT16) {
            //sd::TypeCast::convertGeneric<int16_t, int16_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_UINT16) {
//            sd::TypeCast::convertGeneric<int16_t, uint16_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO...
        } else if (dstType == ND4J_FLOAT32) {
            sd::TypeCast::convertGeneric<int16_t, float>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_DOUBLE) {
            sd::TypeCast::convertGeneric<int16_t, double>(nullptr, hx, N, hz);
        } else {
            printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_FLOAT24) {

    } else if (srcType == ND4J_FLOAT32) {
        if (dstType == ND4J_FLOAT8) {
        //    sd::TypeCast::convertGeneric<float, sd::float8>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_INT8) {
            sd::TypeCast::convertGeneric<float, int8_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_UINT8) {
            sd::TypeCast::convertGeneric<float, uint8_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_FLOAT16) {
            sd::TypeCast::convertGeneric<float, float16>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_INT16) {
            sd::TypeCast::convertGeneric<float, int16_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_UINT16) {
//            sd::TypeCast::convertGeneric<float, uint16_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_DOUBLE) {
            sd::TypeCast::convertGeneric<float, double>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_THRESHOLD) {
            sd::TypeCast::convertToThreshold<float>(nullptr, hx, N, hz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_DOUBLE) {
        if (dstType == ND4J_FLOAT8) {
         //   sd::TypeCast::convertGeneric<double, sd::float8>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_INT8) {
            sd::TypeCast::convertGeneric<double, int8_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_UINT8) {
            sd::TypeCast::convertGeneric<double, uint8_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_FLOAT16) {
            sd::TypeCast::convertGeneric<double, float16>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_INT16) {
            sd::TypeCast::convertGeneric<double, int16_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_UINT16) {
//            sd::TypeCast::convertGeneric<double, uint16_t>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_FLOAT32) {
            sd::TypeCast::convertGeneric<double, float>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_DOUBLE) {
            //
        } else if (dstType == ND4J_THRESHOLD) {
            sd::TypeCast::convertToThreshold<double>(nullptr, hx, N, hz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_THRESHOLD) {
        if (dstType == ND4J_FLOAT16) {
            sd::TypeCast::convertFromThreshold<float16>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_FLOAT32) {
            sd::TypeCast::convertFromThreshold<float>(nullptr, hx, N, hz);
        } else if (dstType == ND4J_DOUBLE) {
            sd::TypeCast::convertFromThreshold<double>(nullptr, hx, N, hz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else {
        nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
}

/*
void fillUtf8String(Nd4jPointer *extraPointers, const char **strings, int numStrings, Nd4jPointer buffer) {
    auto hZ = reinterpret_cast<sd::utf8string**>(buffer);
    for (int e = 0; e < numStrings; e++) {
        hZ[e] = reinterpret_cast<sd::utf8string*>(createUtf8String(extraPointers, strings[e]));
    }
}
 */

Nd4jPointer createUtf8String(Nd4jPointer *extraPointers, const char *string, int length) {
    auto u = new sd::utf8string(string, length);
    return reinterpret_cast<Nd4jPointer>(u);
}

Nd4jLong getUtf8StringLength(Nd4jPointer *extraPointers, Nd4jPointer ptr) {
    return reinterpret_cast<sd::utf8string*>(ptr)->_length;
}
char* getUtf8StringBuffer(Nd4jPointer *extraPointers, Nd4jPointer ptr) {
    return reinterpret_cast<sd::utf8string*>(ptr)->_buffer;
}

void deleteUtf8String(Nd4jPointer *extraPointers, Nd4jPointer ptr) {
    delete(reinterpret_cast<sd::utf8string*>(ptr));
}

template <typename I>
static void  _scatterUpdate(
        Nd4jPointer *extraPointers, int opCode, int numOfSubArrs,
        void* hX, const Nd4jLong* hXShapeInfo, const Nd4jLong* hXOffsets,
        void* dX, const Nd4jLong* dXShapeInfo, const Nd4jLong* dXOffsets,
        void* hY, const Nd4jLong* hYShapeInfo, const Nd4jLong* hYOffsets,
        void* dY, const Nd4jLong* dYShapeInfo, const Nd4jLong* dYOffsets,
        void* vIindexes, const Nd4jLong* hIndicesShapeInfo, void* dIindexes, const Nd4jLong* dIndicesShapeInfo) {

    auto hIindexes = reinterpret_cast<I*>(vIindexes);
        auto func = PRAGMA_THREADS_DO {
            for (int i = 0; i < numOfSubArrs; ++i) {
                int threadIndex = thread_id;
            const auto xIndex = hIindexes[i];
            const bool isOwner = xIndex < numThreads ? threadIndex == xIndex : threadIndex == xIndex % numThreads;

            if (!isOwner)
                continue;

                NDArray inSubArr(reinterpret_cast<int8_t *>(hX) + (hXOffsets[hIindexes[i]] * DataTypeUtils::sizeOf(hXShapeInfo)), hXShapeInfo);
                NDArray updSubArr(reinterpret_cast<int8_t *>(hY) + (hYOffsets[i] * DataTypeUtils::sizeOf(hXShapeInfo)), hYShapeInfo);

            if (inSubArr.lengthOf() != updSubArr.lengthOf()) {
                continue;
            }

                switch (opCode) {
                    case 0:
                        inSubArr.applyPairwiseTransform(pairwise::Add, updSubArr, inSubArr);
                        break;
                    case 1:
                        inSubArr.applyPairwiseTransform(pairwise::Subtract, updSubArr, inSubArr);
                        break;
                    case 2:
                        inSubArr.applyPairwiseTransform(pairwise::Multiply, updSubArr, inSubArr);
                        break;
                    case 3:
                        inSubArr.applyPairwiseTransform(pairwise::Divide, updSubArr, inSubArr);
                        break;
                    case 4:
                        inSubArr.applyPairwiseTransform(pairwise::ReverseSubtract, updSubArr, inSubArr);
                        break;
                    case 5:
                        inSubArr.applyPairwiseTransform(pairwise::ReverseDivide, updSubArr, inSubArr);
                        break;
                    case 6:
                        inSubArr.applyPairwiseTransform(pairwise::CopyPws, updSubArr, inSubArr);
                        break;
                    default:
                        continue;
                }
            }
        };

        samediff::Threads::parallel_do(func);
}

////////////////////////////////////////////////////////////////////////
void scatterUpdate(Nd4jPointer *extraPointers, int opCode, int numOfSubArrs,
                      void* hX, const Nd4jLong* hXShapeInfo, const Nd4jLong* hXOffsets,
                      void* dX, const Nd4jLong* dXShapeInfo, const Nd4jLong* dXOffsets,
                      void* hY, const Nd4jLong* hYShapeInfo, const Nd4jLong* hYOffsets,
                      void* dY, const Nd4jLong* dYShapeInfo, const Nd4jLong* dYOffsets,
                      void* hIindexes, const Nd4jLong* hIndicesShapeInfo, void* dIindexes, const Nd4jLong* dIndicesShapeInfo) {
    auto iType = ArrayOptions::dataType(hIndicesShapeInfo);

    try {
        BUILD_SINGLE_SELECTOR(iType, _scatterUpdate, (extraPointers, opCode, numOfSubArrs, hX, hXShapeInfo, hXOffsets, dX, dXShapeInfo, dXOffsets, hY, hYShapeInfo, hYOffsets, dY, dYShapeInfo, dYOffsets, hIindexes, hIndicesShapeInfo, dIindexes, dIndicesShapeInfo), INDEXING_TYPES);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}


void inspectArray(Nd4jPointer *extraPointers, Nd4jPointer buffer, Nd4jLong *shapeInfo, Nd4jPointer specialBuffer, Nd4jLong *specialShapeInfo, Nd4jPointer debugInfo) {
    try {
        auto p = reinterpret_cast<sd::DebugInfo *>(debugInfo);
        NDArray array(buffer, shapeInfo);
        sd::DebugHelper::retrieveDebugStatistics(p, &array);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void tryPointer(Nd4jPointer extra, Nd4jPointer p, int len) {
    try {
        auto buf = reinterpret_cast<int8_t *>(p);
        int cnt = 0;
        for (int i = 0; i < len; i++)
            cnt += buf[cnt];
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

sd::ConstantShapeBuffer* shapeBuffer(int rank, Nd4jLong *shape, Nd4jLong *strides, sd::DataType dtype, char order, Nd4jLong ews, bool empty) {
    try {
        auto buffer = new ConstantShapeBuffer();
        *buffer = sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(
                ShapeDescriptor(dtype, order, shape, strides, rank, ews, empty));
        return buffer;
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return nullptr;
    }
}

void deleteConstantShapeBuffer(sd::ConstantShapeBuffer* ptr) {
    delete ptr;
}

void deleteConstantDataBuffer(sd::ConstantDataBuffer* ptr) {
  delete ptr;
}

void deleteTadPack(sd::TadPack* ptr) {
    delete ptr;
}

sd::ConstantDataBuffer* constantBufferLong(sd::DataType dtype, const Nd4jLong *data, int length) {
    return nullptr;
}

sd::ConstantDataBuffer* constantBufferDouble(sd::DataType dtype, double *data, int length) {
    return nullptr;
}

sd::ConstantDataBuffer* constantBuffer(sd::DataType dtype, sd::ConstantDescriptor *descriptor) {
    try {
        return sd::ConstantHelper::getInstance().constantBuffer(*descriptor, dtype);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return nullptr;
    }
}

Nd4jPointer getConstantShapeBufferPrimary(sd::ConstantShapeBuffer* dbf) {
  return const_cast<Nd4jLong*>(dbf->primary());
}

Nd4jPointer getConstantShapeBufferSpecial(sd::ConstantShapeBuffer* dbf) {
  return const_cast<Nd4jLong*>(dbf->special());
}

Nd4jPointer getConstantDataBufferPrimary(sd::ConstantDataBuffer* dbf) {
    return dbf->primary();
}
Nd4jPointer getConstantDataBufferSpecial(sd::ConstantDataBuffer* dbf) {
    return dbf->special();
}
Nd4jLong getConstantDataBufferLength(sd::ConstantDataBuffer* dbf) {
    return dbf->length();
}
Nd4jLong getConstantDataBufferSizeOf(sd::ConstantDataBuffer* dbf) {
    return dbf->sizeOf();
}


sd::graph::Context* createGraphContext(int nodeId) {
    try {
        return new sd::graph::Context(nodeId);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return nullptr;
    }
}
sd::graph::RandomGenerator* getGraphContextRandomGenerator(sd::graph::Context* ptr) {
    return &ptr->randomGenerator();
}
void markGraphContextInplace(sd::graph::Context* ptr, bool reallyInplace) {
    ptr->markInplace(reallyInplace);
}
void setGraphContextCudaContext(sd::graph::Context* ptr, void *stream, void *reductionPointer, void *allocationPointer) {
}
void setGraphContextInputArray(sd::graph::Context* ptr, int index, void *buffer, void *shapeInfo, void *specialBuffer, void *specialShapeInfo) {
    ptr->setInputArray(index, buffer, shapeInfo, specialBuffer, specialShapeInfo);
}
void setGraphContextOutputArray(sd::graph::Context* ptr, int index, void *buffer, void *shapeInfo, void *specialBuffer, void *specialShapeInfo) {
    ptr->setOutputArray(index, buffer, shapeInfo, specialBuffer, specialShapeInfo);
}

void setGraphContextInputBuffer(OpaqueContext* ptr, int index, OpaqueDataBuffer *buffer, void *shapeInfo, void *specialShapeInfo) {
    ptr->setInputArray(index, buffer, shapeInfo, specialShapeInfo);
}

void setGraphContextOutputBuffer(OpaqueContext* ptr, int index, OpaqueDataBuffer *buffer, void *shapeInfo, void *specialShapeInfo) {
    ptr->setOutputArray(index, buffer, shapeInfo, specialShapeInfo);
}

void setGraphContextTArguments(sd::graph::Context* ptr, double *arguments, int numberOfArguments) {
    ptr->setTArguments(arguments, numberOfArguments);
}
void setGraphContextIArguments(sd::graph::Context* ptr, Nd4jLong *arguments, int numberOfArguments) {
    ptr->setIArguments(arguments, numberOfArguments);
}
void setGraphContextBArguments(sd::graph::Context* ptr, bool *arguments, int numberOfArguments) {
    ptr->setBArguments(arguments, numberOfArguments);
}

void setGraphContextDArguments(OpaqueContext* ptr, int *arguments, int numberOfArguments) {
    std::vector<sd::DataType> dtypes(numberOfArguments);
    for (int e = 0; e < numberOfArguments; e++)
        dtypes[e] = (sd::DataType) arguments[e];

    ptr->setDArguments(dtypes);
}

void deleteGraphContext(sd::graph::Context* ptr) {
    delete ptr;
}

void ctxAllowHelpers(OpaqueContext* ptr, bool reallyAllow) {
    ptr->allowHelpers(reallyAllow);
}

void ctxSetExecutionMode(OpaqueContext* ptr, int execMode) {
    if (execMode < 0 || execMode > 2)
        execMode = 0;

    ptr->setExecutionMode((samediff::ExecutionMode) execMode);
}

void ctxPurge(OpaqueContext* ptr) {
    ptr->clearFastPath();
}

sd::graph::RandomGenerator* createRandomGenerator(Nd4jLong rootSeed, Nd4jLong nodeSeed) {
    return new sd::graph::RandomGenerator(rootSeed, nodeSeed);
}

Nd4jLong getRandomGeneratorRootState(sd::graph::RandomGenerator* ptr) {
    return ptr->rootState();
}

Nd4jLong getRandomGeneratorNodeState(sd::graph::RandomGenerator* ptr) {
    return ptr->nodeState();
}

void setRandomGeneratorStates(sd::graph::RandomGenerator* ptr, Nd4jLong rootSeed, Nd4jLong nodeSeed) {
    ptr->setStates(rootSeed, nodeSeed);
}

float getRandomGeneratorRelativeFloat(sd::graph::RandomGenerator* ptr, Nd4jLong index) {
  return ptr->relativeT<float>(index);
}

double getRandomGeneratorRelativeDouble(sd::graph::RandomGenerator* ptr, Nd4jLong index) {
  return ptr->relativeT<double>(index);
}

int getRandomGeneratorRelativeInt(sd::graph::RandomGenerator* ptr, Nd4jLong index) {
    return ptr->relativeInt(index);
}

Nd4jLong getRandomGeneratorRelativeLong(sd::graph::RandomGenerator* ptr, Nd4jLong index) {
    return ptr->relativeLong(index);
}

void deleteRandomGenerator(sd::graph::RandomGenerator* ptr) {
    delete ptr;
}


int dataTypeFromNpyHeader(void *header) {
    return (int) cnpy::dataTypeFromHeader(reinterpret_cast<char *>(header));
}

Nd4jPointer shapeBufferForNumpy(Nd4jPointer npyArray) {
    try {
        cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char *>(npyArray));
        unsigned int shapeSize = arr.shape.size();
        std::vector<Nd4jLong> shape(shapeSize);
        bool _empty = false;
        for (unsigned int i = 0; i < shapeSize; i++) {
            shape[i] = arr.shape[i];

            if (arr.shape[i] == 0)
                _empty = true;
        }

        auto dtype = cnpy::dataTypeFromHeader(reinterpret_cast<char *>(npyArray));

        Nd4jLong *shapeBuffer;
        if (shape.size() == 1 && shape[0] == 0) {
            // scalar case
            shapeBuffer = sd::ShapeBuilders::createScalarShapeInfo(dtype);
        } else if (_empty) {
            if (shapeSize > 0)
                shapeBuffer = sd::ShapeBuilders::emptyShapeInfo(dtype, arr.fortranOrder ? 'f' : 'c', shape);
            else
                shapeBuffer = sd::ShapeBuilders::emptyShapeInfo(dtype);
        } else {
            shapeBuffer = sd::ShapeBuilders::createShapeInfo(dtype, arr.fortranOrder ? 'f' : 'c', shape);
        }
        return const_cast<Nd4jLong*>(sd::ConstantShapeHelper::getInstance().createFromExisting(shapeBuffer, true));
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return nullptr;
    }
}

void sortByKey(Nd4jPointer *extraPointers,
                          void *x, const Nd4jLong *xShapeInfo,
                          void *dx, const Nd4jLong *dxShapeInfo,
                          void *y, const Nd4jLong *yShapeInfo,
                          void *dy, const Nd4jLong *dyShapeInfo,
                          bool descending) {
    try {
        auto xType = ArrayOptions::dataType(xShapeInfo);
        auto yType = ArrayOptions::dataType(yShapeInfo);

        BUILD_DOUBLE_SELECTOR(xType, yType, sd::DoubleMethods, ::sortByKey(x, xShapeInfo, y, yShapeInfo, descending), LIBND4J_TYPES, LIBND4J_TYPES);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void sortByValue(Nd4jPointer *extraPointers,
                            void *x, const Nd4jLong *xShapeInfo,
                            void *dx, const Nd4jLong *dxShapeInfo,
                            void *y, const Nd4jLong *yShapeInfo,
                            void *dy, const Nd4jLong *dyShapeInfo,
                            bool descending) {
    try {
        auto xType = ArrayOptions::dataType(xShapeInfo);
        auto yType = ArrayOptions::dataType(yShapeInfo);

        BUILD_DOUBLE_SELECTOR(xType, yType, sd::DoubleMethods, ::sortByValue(x, xShapeInfo, y, yShapeInfo, descending), LIBND4J_TYPES, LIBND4J_TYPES);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void sortTadByKey(Nd4jPointer *extraPointers,
                  void *x, const Nd4jLong *xShapeInfo,
                  void *dx, const Nd4jLong *dxShapeInfo,
                  void *y, const Nd4jLong *yShapeInfo,
                  void *dy, const Nd4jLong *dyShapeInfo,
                  int *dimension, int dimensionLength,
                  bool descending) {
    try {
        auto xType = ArrayOptions::dataType(xShapeInfo);
        auto yType = ArrayOptions::dataType(yShapeInfo);

        BUILD_DOUBLE_SELECTOR(xType, yType, sd::DoubleMethods, ::sortTadByKey(x, xShapeInfo, y, yShapeInfo, dimension, dimensionLength, descending), LIBND4J_TYPES, LIBND4J_TYPES);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

void sortTadByValue(Nd4jPointer *extraPointers,
                    void *x, const Nd4jLong *xShapeInfo,
                    void *dx, const Nd4jLong *dxShapeInfo,
                    void *y, const Nd4jLong *yShapeInfo,
                    void *dy, const Nd4jLong *dyShapeInfo,
                    int *dimension, int dimensionLength,
                    bool descending) {
    try {
        auto xType = ArrayOptions::dataType(xShapeInfo);
        auto yType = ArrayOptions::dataType(yShapeInfo);

        BUILD_DOUBLE_SELECTOR(xType, yType, sd::DoubleMethods, ::sortTadByValue(x, xShapeInfo, y, yShapeInfo, dimension, dimensionLength, descending), LIBND4J_TYPES, LIBND4J_TYPES);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

const char* runLightBenchmarkSuit(bool printOut) {
    try {
        sd::LightBenchmarkSuit suit;
        auto result = suit.runSuit();

        if (printOut)
            nd4j_printf("%s\n", result.data());

        auto chars = new char[result.length() + 1];
        std::memcpy(chars, result.data(), result.length());
        chars[result.length()] = (char) 0x0;

        return chars;
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return nullptr;
    }
}

Nd4jLong getCachedMemory(int deviceId) {
    return sd::ConstantHelper::getInstance().getCachedAmount(deviceId);
}

const char* runFullBenchmarkSuit(bool printOut) {
    try {
        sd::FullBenchmarkSuit suit;
        auto result = suit.runSuit();

        if (printOut)
            nd4j_printf("%s\n", result.data());

        auto chars = new char[result.length() + 1];
        std::memcpy(chars, result.data(), result.length());
        chars[result.length()] = (char) 0x0;

        return chars;
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return nullptr;
    }
}

sd::LaunchContext* defaultLaunchContext() {
    return LaunchContext::defaultContext();
}

Nd4jPointer lcScalarPointer(OpaqueLaunchContext* lc) {
    return nullptr;
}

Nd4jPointer lcReductionPointer(OpaqueLaunchContext* lc) {
    return nullptr;
}

Nd4jPointer lcAllocationPointer(OpaqueLaunchContext* lc) {
    return nullptr;
}

Nd4jPointer lcExecutionStream(OpaqueLaunchContext* lc) {
    return nullptr;
}

Nd4jPointer lcCopyStream(OpaqueLaunchContext* lc) {
    return nullptr;
}

Nd4jPointer lcBlasHandle(OpaqueLaunchContext* lc) {
    return nullptr;
}

Nd4jPointer lcSolverHandle(OpaqueLaunchContext* lc) {
    return nullptr;
}

int lastErrorCode() {
    return sd::LaunchContext::defaultContext()->errorReference()->errorCode();
}

const char* lastErrorMessage() {
    return sd::LaunchContext::defaultContext()->errorReference()->errorMessage();
}

void ctxShapeFunctionOverride(OpaqueContext* ptr, bool reallyOverride) {
    ptr->setShapeFunctionOverride(reallyOverride);
}

int  binaryLevel() {
#ifdef CPU_FEATURES

#if defined(F_X64)
    return 1;
#elif defined (F_AVX2)
    return 2;
#elif defined (F_AVX512)
    return 3;
#else
    return 0;
#endif

#else
    return 0;
#endif
}

int optimalLevel() {
#ifdef CPU_FEATURES
    auto features = cpu_features::GetX86Info().features;

    if (features.avx && features.avx2 && features.avx512f && features.avx512vl && features.avx512bw && features.avx512dq && features.avx512cd)
        return 3;
    else if (features.avx && features.avx2)
        return 2;
    else
        return 1;

#else
    return 0;
#endif
}

bool isMinimalRequirementsMet() {
#ifdef CPU_FEATURES
    auto features = cpu_features::GetX86Info().features;

#if defined(F_X64)
    return true;
#elif defined (F_AVX2)
    return features.avx && features.avx2;
#elif defined (F_AVX512)
    // we're optimizing for skylake-avx512 features, so we'll check those out
    return features.avx && features.avx2 && features.avx512f && features.avx512vl && features.avx512bw && features.avx512dq && features.avx512cd;
#else
    return true;
#endif

#else
    return true;
#endif
}

bool isOptimalRequirementsMet() {
#ifdef CPU_FEATURES
    auto b = ::binaryLevel();
    auto o = ::optimalLevel();

    if (b == o)
        return true;
    else
        return false;
#else
    return true;
#endif
}

OpaqueDataBuffer* dbAllocateDataBuffer(Nd4jLong elements, int dataType, bool allocateBoth) {
    return allocateDataBuffer(elements, dataType, allocateBoth);
}

OpaqueDataBuffer* allocateDataBuffer(Nd4jLong elements, int dataType, bool allocateBoth) {
    try {
        auto dtype = DataTypeUtils::fromInt(dataType);
        return new sd::InteropDataBuffer(elements * DataTypeUtils::sizeOf(dtype) , dtype, allocateBoth);
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
        return nullptr;
    }
}

Nd4jPointer dbPrimaryBuffer(OpaqueDataBuffer *dataBuffer) {
    return dataBuffer->primary();
}

Nd4jPointer dbSpecialBuffer(OpaqueDataBuffer *dataBuffer) {
    return dataBuffer->special();
}

void deleteDataBuffer(OpaqueDataBuffer *dataBuffer) {
    delete dataBuffer;
}

OpaqueDataBuffer* dbCreateExternalDataBuffer(Nd4jLong elements, int dataType, Nd4jPointer primary, Nd4jPointer special) {
    auto buffer = dbAllocateDataBuffer(0, dataType, false);

    if (primary != nullptr)
        buffer->setPrimary(primary, elements);

    if (special != nullptr)
        buffer->setSpecial(special, elements);

    return buffer;
}

void dbSetPrimaryBuffer(OpaqueDataBuffer *dataBuffer, Nd4jPointer primaryBuffer, Nd4jLong numBytes) {
    dataBuffer->setPrimary(primaryBuffer, numBytes);
}

void dbSetSpecialBuffer(OpaqueDataBuffer *dataBuffer, Nd4jPointer specialBuffer, Nd4jLong numBytes) {
    dataBuffer->setSpecial(specialBuffer, numBytes);
}

void dbAllocatePrimaryBuffer(OpaqueDataBuffer *dataBuffer) {
    dataBuffer->dataBuffer()->allocatePrimary();
}

void dbAllocateSpecialBuffer(OpaqueDataBuffer *dataBuffer) {
    dataBuffer->dataBuffer()->allocateSpecial();
}

void dbExpandBuffer(OpaqueDataBuffer *dataBuffer, Nd4jLong elements) {
    try {
        dataBuffer->dataBuffer()->expand(elements * DataTypeUtils::sizeOf(dataBuffer->dataBuffer()->getDataType()));
    } catch (std::exception &e) {
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    }
}

OpaqueDataBuffer* dbCreateView(OpaqueDataBuffer *dataBuffer, Nd4jLong length, Nd4jLong offset) {
    return new InteropDataBuffer(*dataBuffer, length, offset);
}

void dbSyncToSpecial(OpaqueDataBuffer *dataBuffer) {
    dataBuffer->dataBuffer()->syncToSpecial();
}

void dbSyncToPrimary(OpaqueDataBuffer *dataBuffer) {
    dataBuffer->dataBuffer()->syncToPrimary(nullptr);
}

void dbTickHostRead(OpaqueDataBuffer *dataBuffer) {
    dataBuffer->dataBuffer()->readPrimary();
}

void dbTickHostWrite(OpaqueDataBuffer *dataBuffer) {
    dataBuffer->dataBuffer()->writePrimary();
}

void dbTickDeviceRead(OpaqueDataBuffer *dataBuffer) {
    dataBuffer->dataBuffer()->readSpecial();
}

void dbTickDeviceWrite(OpaqueDataBuffer *dataBuffer) {
    dataBuffer->dataBuffer()->writeSpecial();
}

void dbExpand(OpaqueDataBuffer *dataBuffer, Nd4jLong elements) {
    dataBuffer->expand(elements);
}

int dbLocality(OpaqueDataBuffer *dataBuffer) {
    return 0;
}

void dbSetDeviceId(OpaqueDataBuffer *dataBuffer, int deviceId) {
    dataBuffer->setDeviceId(deviceId);
}

int dbDeviceId(OpaqueDataBuffer *dataBuffer) {
    return dataBuffer->deviceId();
}

void dbClose(OpaqueDataBuffer *dataBuffer) {
    dataBuffer->getDataBuffer()->close();
}

BUILD_SINGLE_TEMPLATE(template void pullRowsGeneric, (void *, Nd4jLong const*, void*, Nd4jLong const*, const int, Nd4jLong const*, Nd4jLong const*, Nd4jLong const*, Nd4jLong const*, Nd4jLong const*), LIBND4J_TYPES);
BUILD_SINGLE_TEMPLATE(template void tearGeneric, (void *, Nd4jLong const* , Nd4jPointer*, Nd4jLong const*, Nd4jLong const*, Nd4jLong const*), LIBND4J_TYPES);
BUILD_SINGLE_TEMPLATE(template void shuffleGeneric, (void**, Nd4jLong* const*, void**, Nd4jLong* const*, int, int*, Nd4jLong* const*, Nd4jLong* const*), LIBND4J_TYPES);


