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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 14.03.2019
//

#include <helpers/Loops.h>

using namespace simdOps;


//////////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
void nd4j::IndexReductionLoops<X>::loopIndexReduce(X* x, Nd4jLong* xShapeInfo,
                           Nd4jLong* z, Nd4jLong* zShapeInfo,
                           Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets,
                           X* extraParams) {

    LoopKind kindOfLoop = nd4j::ReductionLoops<X,X,X>::deduceKindOfLoopTadXZ(xShapeInfo, zShapeInfo, tadShapeInfo);
    if(kindOfLoop == SMALLARR2DX)
        kindOfLoop = EWSNONZERO;

    const Nd4jLong zLen   = shape::length(zShapeInfo);
    const Nd4jLong tadLen = shape::length(tadShapeInfo);

    const uint tadEws = shape::elementWiseStride(tadShapeInfo);
    const uint zEws   = shape::elementWiseStride(zShapeInfo);

    const Nd4jLong* tadShape  = shape::shapeOf(const_cast<Nd4jLong*>(tadShapeInfo));
    const Nd4jLong* tadStride = shape::stride(const_cast<Nd4jLong*>(tadShapeInfo));

    int tadsPerThread = zLen / TAD_THRESHOLD;
    int numThreads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
    numThreads = nd4j::math::nd4j_min<int>(numThreads, omp_get_max_threads());

    switch (kindOfLoop) {
        //*********************************************//
        case EWS1: {

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; i++) {
                auto tad = const_cast<X*>(x) + tadOffsets[i];
                auto indexValue = OpType::startingIndexValue(tad);

                for (uint j = 0; j < tadLen; j++) {
                    functions::indexreduce::IndexValue<X> comp(tad[j], j);
                    indexValue = OpType::update(indexValue, comp, extraParams);
                }

                z[i] = indexValue.index;
            }
        }
            break;

            //*********************************************//
        case EWSNONZERO: {

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; i++) {
                auto tad = const_cast<X*>(x) + tadOffsets[i];
                auto indexValue = OpType::startingIndexValue(tad);

                for (uint j = 0; j < tadLen; j++) {
                    functions::indexreduce::IndexValue<X> comp(tad[j * tadEws], j);
                    indexValue = OpType::update(indexValue, comp, extraParams);
                }

                z[i * zEws] = indexValue.index;
            }
        }
            break;

            //*********************************************//
        case RANK1: {

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; i++) {
                auto tad = const_cast<X*>(x) + tadOffsets[i];
                auto indexValue = OpType::startingIndexValue(tad);

                for (uint i0 = 0; i0 < tadLen; ++i0) {
                    functions::indexreduce::IndexValue<X> comp(tad[i0 * tadStride[0]], i0);
                    indexValue = OpType::update(indexValue, comp, extraParams);
                }

                z[i] = indexValue.index;
            }
        }
            break;

            //*********************************************//
        case RANK2: {
            Nd4jLong newStride[2];
            shape::updateStrides(2, tadShape, newStride, 'c');

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; ++i) {
                auto tad = const_cast<X*>(x) + tadOffsets[i];
                auto indexValue = OpType::startingIndexValue(tad);

                for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                    for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                        const auto tadOffset = i0 * tadStride[0] + i1 * tadStride[1];
                        const auto tadIndex  = i0 * newStride[0] + i1;
                        functions::indexreduce::IndexValue<X> comp(tad[tadOffset], tadIndex);
                        indexValue = OpType::update(indexValue, comp, extraParams);
                    }
                }

                z[i] = indexValue.index;
            }
        }
            break;

            //*********************************************//
        case RANK3: {
            Nd4jLong newStride[3];
            shape::updateStrides(3, tadShape, newStride, 'c');

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; ++i) {
                auto tad = const_cast<X*>(x) + tadOffsets[i];
                auto indexValue = OpType::startingIndexValue(tad);

                for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                    for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                        for (uint i2 = 0; i2 < tadShape[2]; ++i2) {
                            const auto tadOffset = i0 * tadStride[0] + i1 * tadStride[1] + i2 * tadStride[2];
                            const auto tadIndex  = i0 * newStride[0] + i1 * newStride[1] + i2;
                            functions::indexreduce::IndexValue<X> comp(tad[tadOffset], tadIndex);
                            indexValue = OpType::update(indexValue, comp, extraParams);
                        }
                    }
                }

                z[i] = indexValue.index;
            }
        }
            break;

            //*********************************************//
        case RANK4: {
            Nd4jLong newStride[4];
            shape::updateStrides(4, tadShape, newStride, 'c');

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; ++i) {
                auto tad = const_cast<X*>(x) + tadOffsets[i];
                auto indexValue = OpType::startingIndexValue(tad);

                for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                    for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                        for (uint i2 = 0; i2 < tadShape[2]; ++i2) {
                            for (uint i3 = 0; i3 < tadShape[3]; ++i3) {
                                const auto tadOffset = i0 * tadStride[0] + i1 * tadStride[1] + i2 * tadStride[2] + i3 * tadStride[3];
                                const auto tadIndex  = i0 * newStride[0] + i1 * newStride[1] + i2 * newStride[2] + i3;
                                functions::indexreduce::IndexValue<X> comp(tad[tadOffset], tadIndex);
                                indexValue = OpType::update(indexValue, comp, extraParams);
                            }
                        }
                    }
                }

                z[i] = indexValue.index;
            }
        }
            break;

            //*********************************************//
        case RANK5: {
            Nd4jLong newStride[5];
            shape::updateStrides(5, tadShape, newStride, 'c');

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; ++i) {
                auto tad = const_cast<X*>(x) + tadOffsets[i];
                auto indexValue = OpType::startingIndexValue(tad);

                for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                    for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                        for (uint i2 = 0; i2 < tadShape[2]; ++i2) {
                            for (uint i3 = 0; i3 < tadShape[3]; ++i3) {
                                for (uint i4 = 0; i4 < tadShape[4]; ++i4) {
                                    const auto tadOffset = i0 * tadStride[0] + i1 * tadStride[1] + i2 * tadStride[2] + i3 * tadStride[3] + i4 * tadStride[4];
                                    const auto tadIndex  = i0 * newStride[0] + i1 * newStride[1] + i2 * newStride[2] + i3 * newStride[3] + i4;
                                    functions::indexreduce::IndexValue<X> comp(tad[tadOffset], tadIndex);
                                    indexValue = OpType::update(indexValue, comp, extraParams);
                                }
                            }
                        }
                    }
                }

                z[i] = indexValue.index;
            }
        }
            break;

            //*********************************************//
        case X_EWSNONZERO: {
            uint castZShapeInfo[MAX_RANK];
            const bool canCastZ   = nd4j::DataTypeUtils::castShapeInfo<uint>(zShapeInfo,   castZShapeInfo);

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; i++) {
                auto tad = const_cast<X*>(x) + tadOffsets[i];
                auto indexValue = OpType::startingIndexValue(tad);

                for (uint j = 0; j < tadLen; j++) {
                    functions::indexreduce::IndexValue<X> comp(tad[j * tadEws], j);
                    indexValue = OpType::update(indexValue, comp, extraParams);
                }

                auto zOffset = shape::indexOffset(i, zShapeInfo, castZShapeInfo, zLen, canCastZ);
                z[zOffset] = indexValue.index;
            }
        }
            break;

            //*********************************************//
        case Z_EWSNONZERO: {
            uint castTadShapeInfo[MAX_RANK];
            const bool canCastTad = nd4j::DataTypeUtils::castShapeInfo<uint>(tadShapeInfo, castTadShapeInfo);

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; i++) {
                auto tad = const_cast<X*>(x) + tadOffsets[i];
                auto indexValue = OpType::startingIndexValue(tad);

                for (uint j = 0; j < tadLen; j++) {
                    auto tadOffset = shape::indexOffset(j, tadShapeInfo, castTadShapeInfo, tadLen, canCastTad);
                    functions::indexreduce::IndexValue<X> comp(tad[tadOffset], j);
                    indexValue = OpType::update(indexValue, comp, extraParams);
                }

                z[i * zEws] = indexValue.index;
            }
        }
            break;

            //*********************************************//
        default: {
            uint castTadShapeInfo[MAX_RANK];
            uint castZShapeInfo[MAX_RANK];
            const bool canCastTad = nd4j::DataTypeUtils::castShapeInfo<uint>(tadShapeInfo, castTadShapeInfo);
            const bool canCastZ   = nd4j::DataTypeUtils::castShapeInfo<uint>(zShapeInfo,   castZShapeInfo);

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; i++) {
                auto tad = const_cast<X*>(x) + tadOffsets[i];
                auto indexValue = OpType::startingIndexValue(tad);

                for (uint j = 0; j < tadLen; j++) {
                    auto tadOffset = shape::indexOffset(j, tadShapeInfo, castTadShapeInfo, tadLen, canCastTad);
                    functions::indexreduce::IndexValue<X> comp(tad[tadOffset], j);
                    indexValue = OpType::update(indexValue, comp, extraParams);
                }

                auto zOffset = shape::indexOffset(i, zShapeInfo, castZShapeInfo, zLen, canCastZ);
                z[zOffset] = indexValue.index;
            }
        }
    }
}

template <typename X>
void nd4j::IndexReductionLoops<X>::wrapIndexReduce(const int opNum, void* vx, Nd4jLong* xShapeInfo, Nd4jLong* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, void* vextraParams) {
    auto x = reinterpret_cast<X *>(vx);
    auto extraParams = reinterpret_cast<X *>(vextraParams);

    DISPATCH_BY_OPNUM_T(loopIndexReduce, PARAMS(x, xShapeInfo, z, zShapeInfo, tadShapeInfo, tadOffsets, extraParams), INDEX_REDUCE_OPS);
}

BUILD_SINGLE_TEMPLATE(template void nd4j::IndexReductionLoops, ::wrapIndexReduce(const int opNum, void* vx, Nd4jLong* xShapeInfo, Nd4jLong* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, void* vextraParams), LIBND4J_TYPES);