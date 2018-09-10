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
// Created by raver119 on 29/10/17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_ismax)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(ismax, 1, 1, true, 0, -1) {
            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);
            std::vector<int> dimensions = *(block.getIArguments());			// argI

            // FIXME: this should be moved to helpers!
            if (x->isVector()) {
                int dimensionsLength = dimensions.size();
                int length = x->lengthOf();
                if ((x->shapeOf())[dimensions[0]] == 1) {
                    for (int i = 0; i < length; i++)
                        z->putScalar(i, 1.f);
                }
                else {
                    int eleStride = shape::elementWiseStride(x->getShapeInfo());
                    if (eleStride == 1) {
                        int maxIdx = 0;
                        T currMax = x->getScalar(0);
                        if (length < ELEMENT_THRESHOLD) {

//#pragma omp simd reduction(max:maxIdx,currMax)
                            for (int i = 0; i < length; i++) {
                                if (currMax < x->getScalar(i)) {
                                    currMax = x->getScalar(i);
                                    maxIdx = i;
                                }
                                x->putScalar(i, 0.f);
                            }
                        }
                        else {
#pragma omp parallel proc_bind(AFFINITY) default(shared)
                            {
                                int maxIdxLocal = maxIdx;
                                T currMaxLocal = currMax;
//#pragma omp simd reduction(max:maxIdxLocal,currMaxLocal)
                                for (int i = 0; i < length; i++) {
                                    if (currMaxLocal < x->getScalar(i)) {
                                        currMaxLocal = x->getScalar(i);
                                        maxIdxLocal = i;
                                    }
                                    z->putScalar(i, 0.f);
                                }
#pragma omp critical
                                {
                                    if (currMax < currMaxLocal) {
                                        currMax = currMaxLocal;
                                        maxIdx = maxIdxLocal;
                                    }
                                }
                            }
                        }
                        z->putScalar(maxIdx, 1.f);
                    }
                    else {
                        int maxIdx = 0;
                        T currMax = x->getScalar(0);
                        if (length < ELEMENT_THRESHOLD) {
//#pragma omp parallel for reduction(max:maxIdx,currMax) proc_bind(AFFINITY)
                            for (int i = 0; i < length; i++) {
                                if (currMax < x->getScalar(i*eleStride)) {
                                    currMax = x->getScalar(i*eleStride);
                                    maxIdx = i;
                                }
                                z->putScalar(i, 0.f);
                            }
                        }
                        else {
#pragma omp parallel proc_bind(AFFINITY) default(shared)
                            {
                                int maxIdxLocal = maxIdx;
                                T currMaxLocal = currMax;
//#pragma omp parallel for reduction(max:maxIdx,currMax)  proc_bind(AFFINITY)
                                for (int i = 0; i < length; i++) {
                                    if (currMaxLocal < x->getScalar(i*eleStride)) {
                                        currMaxLocal = x->getScalar(i*eleStride);
                                        maxIdxLocal = i;
                                    }
                                    z->putScalar(i, 0.f);
                                }
#pragma omp critical
                                {
                                    if (currMax < currMaxLocal) {
                                        currMax = currMaxLocal;
                                        maxIdx = maxIdxLocal;
                                    }
                                }
                            }
                        }
                        z->putScalar(maxIdx, 1.f);
                    }
                }
            }
            else {
                int dimensionsLength = dimensions.size();
//                int tads = tad.numTads;
                //decompose in to several sub tads after
                //moving all dimensions (in sorted order)
                //to the back.
                //permuted version of the x shape info for setting up the tad problem
                shape::TAD tad(x->getShapeInfo(), dimensions.data(), dimensionsLength);
                tad.createTadOnlyShapeInfo();
                tad.createOffsets();

                Nd4jLong *tadShapeShapeInfo = tad.tadOnlyShapeInfo;
                Nd4jLong* tadOffsets = tad.tadOffsets;

                int tadLength = shape::tadLength(x->getShapeInfo(), dimensions.data(), dimensionsLength);
                int tads = x->lengthOf() / tadLength;

                int tadsPerThread = tads / TAD_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                auto tadEWS = shape::elementWiseStride(tadShapeShapeInfo);
                auto zEWS = tadEWS;

                int span = (tads / num_threads) + 8;

#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY)
                {
                    int tid = omp_get_thread_num();
                    int start = span * tid;
                    int end = span * (tid + 1);
                    if (end > tads) end = tads;

                    for (int r = start; r < end; r++) {
                        if (tadEWS > 0 && zEWS > 0 && dimensionsLength == 1) {
                            T *rX = x->getBuffer() + tadOffsets[r];
                            T *rZ = z->getBuffer() + tadOffsets[r];

                            T maxValue = rX[0];
                            int maxIdx = 0;
                            if (tadEWS == 1 && zEWS == 1) {
//#pragma omp simd reduction(max:maxValue,maxIdx)
                                for (int i = 0; i < tadLength; i++) {
                                    if (rX[i] > maxValue) {
                                        maxIdx = i;
                                        maxValue = rX[i];
                                    }
                                }

#pragma omp simd
                                for (int i = 0; i < tadLength; i++) {
                                    rZ[i] = maxIdx == i ? (T) 1.0 : (T) 0.0;
                                }

                            } else {

//#pragma omp parallel for reduction(max:maxValue,maxIdx) default(shared)
                                for (int i = 0; i < tadLength; i++) {
                                    if (rX[i * tadEWS] > maxValue) {
                                        maxIdx = i;
                                        maxValue = rX[i * tadEWS];
                                    }
                                }

#pragma omp simd
                                for (int i = 0; i < tadLength; i++) {
                                    rZ[i * zEWS] = maxIdx == i ? (T) 1.0 : (T) 0.0;
                                }
                            }
                        } else {
                            int tadsPerThread = tads / TAD_THRESHOLD;
                            int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                            num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                            Nd4jLong offset = tadOffsets[r];
                            Nd4jLong shapeIter[MAX_RANK];
                            Nd4jLong coord[MAX_RANK];
                            int dim;
                            Nd4jLong xStridesIter[MAX_RANK];
                            Nd4jLong resultStridesIter[MAX_RANK];
                            Nd4jLong *xShape = shape::shapeOf(tadShapeShapeInfo);
                            Nd4jLong *xStride = shape::stride(tadShapeShapeInfo);
                            Nd4jLong *resultStride = shape::stride(tadShapeShapeInfo);
                            int rank = shape::rank(tadShapeShapeInfo);
                            T *xPointer = x->getBuffer() + offset;
                            T *resultPointer = z->getBuffer() + offset;
                            T maxValue = xPointer[0];

                            T *maxCursor = resultPointer;
                            Nd4jPointer maxCursorLong = reinterpret_cast<Nd4jPointer>(maxCursor);
                            if (PrepareTwoRawArrayIter<T>(rank,
                                                          xShape,
                                                          xPointer,
                                                          xStride,
                                                          resultPointer,
                                                          resultStride,
                                                          &rank,
                                                          shapeIter,
                                                          &xPointer,
                                                          xStridesIter,
                                                          &resultPointer,
                                                          resultStridesIter) >= 0) {
                                ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                        if (maxValue < xPointer[0]) {
                                            maxCursor = resultPointer;
                                            maxCursorLong = reinterpret_cast<Nd4jPointer>(resultPointer);
                                            maxValue = xPointer[0];
                                        }

                                        resultPointer[0] = 0.0;
                                    }
                                ND4J_RAW_ITER_TWO_NEXT(dim,
                                                       rank,
                                                       coord,
                                                       shapeIter,
                                                       xPointer,
                                                       xStridesIter,
                                                       resultPointer,
                                                       resultStridesIter);
                                maxCursor = reinterpret_cast<T *>(maxCursorLong);
                                maxCursor[0] = 1.0;
                            }
                        }
                    }
                }
            }
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(IsMax, ismax);
    }
}

#endif