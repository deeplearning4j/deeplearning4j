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
// @author Yurii Shyrma, created on 21.09.2018
// @author raver119@gmail.com
//


#include <helpers/TAD.h>
#include<ops/declarable/helpers/ismax.h>
#include <helpers/ConstantTadHelper.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {

template <typename X, typename Z>
static void ismax_(const NDArray* input, NDArray* output, const std::vector<int>& dimensions) {
                        
    if (input->isVector()) {
        int dimensionsLength = dimensions.size();
        int length = input->lengthOf();
        if (!dimensions.empty() && (input->shapeOf())[dimensions[0]] == 1) {
            for (int i = 0; i < length; i++)
                output->p<Z>(i, 1);
        }
        else {
            int eleStride = shape::elementWiseStride(input->getShapeInfo());
            if (eleStride == 1) {
                int maxIdx = 0;
                auto currMax = input->e<X>(0);
                if (length < ELEMENT_THRESHOLD) {

                    for (int i = 0; i < length; i++) {
                        if (currMax < input->e<X>(i)) {
                            currMax = input->e<X>(i);
                            maxIdx = i;
                        }
                        output->p<Z>(i, 0);
                    }
                }
                else {

                    {
                        int maxIdxLocal = maxIdx;
                        auto currMaxLocal = currMax;

                        for (int i = 0; i < length; i++) {
                            if (currMaxLocal < input->e<X>(i)) {
                                currMaxLocal = input->e<X>(i);
                                maxIdxLocal = i;
                            }
                            output->p<Z>(i, 0);
                        }

                        PRAGMA_OMP_CRITICAL
                        {
                            if (currMax < currMaxLocal) {
                                currMax = currMaxLocal;
                                maxIdx = maxIdxLocal;
                            }
                        }
                    }
                }
                output->p<Z>(maxIdx, 1);
            }
            else {
                int maxIdx = 0;
                auto currMax = input->e<X>(0);
                if (length < ELEMENT_THRESHOLD) {

                    for (int i = 0; i < length; i++) {
                        if (currMax < input->e<X>(i*eleStride)) {
                            currMax = input->e<X>(i*eleStride);
                            maxIdx = i;
                        }
                        output->p<Z>(i, 0.f);
                    }
                }
                else {

                    {
                        int maxIdxLocal = maxIdx;
                        auto currMaxLocal = currMax;
                        for (int i = 0; i < length; i++) {
                            if (currMaxLocal < input->e<X>(i*eleStride)) {
                                currMaxLocal = input->e<X>(i*eleStride);
                                       maxIdxLocal = i;
                            }
                            output->p<Z>(i, 0.f);
                        }

                        PRAGMA_OMP_CRITICAL
                        {
                            if (currMax < currMaxLocal) {
                                currMax = currMaxLocal;
                                maxIdx = maxIdxLocal;
                            }
                        }
                    }
                }
                output->p<Z>(maxIdx, 1);
            }
        }
    }
    else {
        int dimensionsLength = dimensions.size();
        //int tads = tad.numTads;
        //decompose in to several sub tads after
        //moving all dimensions (in sorted order)
        //to the back.
        //permuted version of the input shape info for setting up the tad problem
        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), const_cast<int*>(dimensions.data()), dimensionsLength);

        auto tadShapeShapeInfo = tadPack.primaryShapeInfo();
        auto tadOffsets = tadPack.primaryOffsets();

        int tadLength = shape::length(tadShapeShapeInfo);
        int tads = tadPack.numberOfTads();

        int tadsPerThread = tads / TAD_THRESHOLD;
        int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
        num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

        auto tadEWS = shape::elementWiseStride(tadShapeShapeInfo);
        auto zEWS = tadEWS;

        int span = (tads / num_threads) + 8;

        PRAGMA_OMP_PARALLEL_THREADS(num_threads)
        {
            int tid = omp_get_thread_num();
            int start = span * tid;
            int end = span * (tid + 1);
            if (end > tads) end = tads;

            for (int r = start; r < end; r++) {
                if (tadEWS > 0 && zEWS > 0 && dimensionsLength == 1) {
                    auto rX = const_cast<NDArray*>(input)->bufferAsT<X>() + tadOffsets[r];
                    auto rZ = output->bufferAsT<Z>() + tadOffsets[r];

                    auto maxValue = rX[0];
                    int maxIdx = 0;
                    if (tadEWS == 1 && zEWS == 1) {
                        for (int i = 0; i < tadLength; i++) {
                            if (rX[i] > maxValue) {
                                maxIdx = i;
                                maxValue = rX[i];
                            }
                        }

                        PRAGMA_OMP_SIMD
                        for (int i = 0; i < tadLength; i++) {
                            rZ[i] = maxIdx == i ? (Z) 1 : (Z) 0;
                        }
                    } 
                    else {
                        for (int i = 0; i < tadLength; i++) {
                            if (rX[i * tadEWS] > maxValue) {
                                maxIdx = i;
                                maxValue = rX[i * tadEWS];
                            }
                        }

                        PRAGMA_OMP_SIMD
                        for (int i = 0; i < tadLength; i++) {
                            rZ[i * zEWS] = maxIdx == i ? (Z) 1 : (Z) 0;
                        }
                    }
                } 
                else {
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
                    auto xPointer = const_cast<NDArray*>(input)->bufferAsT<X>() + offset;
                    auto resultPointer = output->bufferAsT<Z>() + offset;
                    auto maxValue = xPointer[0];

                    auto maxCursor = resultPointer;
                    Nd4jPointer maxCursorLong = reinterpret_cast<Nd4jPointer>(maxCursor);
                    
                    if (PrepareTwoRawArrayIter<X,Z>(rank, xShape, xPointer, xStride, resultPointer, resultStride, &rank, shapeIter, &xPointer, xStridesIter, &resultPointer, resultStridesIter) >= 0) {
                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); 
                        {
                            if (maxValue < xPointer[0]) {
                                maxCursor = resultPointer;
                                maxCursorLong = reinterpret_cast<Nd4jPointer>(resultPointer);
                                maxValue = xPointer[0];
                            }
                            resultPointer[0] = (Z) 0;
                        }
                        ND4J_RAW_ITER_TWO_NEXT(dim, rank, coord, shapeIter, xPointer, xStridesIter, resultPointer, resultStridesIter);
                        maxCursor = reinterpret_cast<Z*>(maxCursorLong);
                        maxCursor[0] = (Z) 1;
                    }
                }
            }
        }
    }
}


void ismax(nd4j::LaunchContext * context, const NDArray *input, NDArray *output, const std::vector<int>& dimensions) {
    BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), ismax_, (input, output, dimensions), LIBND4J_TYPES, LIBND4J_TYPES);
}


}
}
}

