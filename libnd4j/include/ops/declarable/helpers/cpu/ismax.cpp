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


#include<ops/declarable/helpers/ismax.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {

template <typename T>
static void ismax_(const NDArray* input, NDArray* output, const std::vector<int>& dimensions) {
                        
    if (input->isVector()) {
        int dimensionsLength = dimensions.size();
        int length = input->lengthOf();
        if ((input->shapeOf())[dimensions[0]] == 1) {
            for (int i = 0; i < length; i++)
                output->putScalar<T>(i, 1.f);
        }
        else {
            int eleStride = shape::elementWiseStride(input->getShapeInfo());
            if (eleStride == 1) {
                int maxIdx = 0;
                T currMax = input->getScalar<T>(0);
                if (length < ELEMENT_THRESHOLD) {
//#pragma omp simd reduction(max:maxIdx,currMax)
                    for (int i = 0; i < length; i++) {
                        if (currMax < input->getScalar<T>(i)) {
                            currMax = input->getScalar<T>(i);
                            maxIdx = i;
                        }
                        output->putScalar<T>(i, 0.f);
                    }
                }
                else {
#pragma omp parallel proc_bind(AFFINITY) default(shared)
                    {
                        int maxIdxLocal = maxIdx;
                        T currMaxLocal = currMax;
//#pragma omp simd reduction(max:maxIdxLocal,currMaxLocal)
                        for (int i = 0; i < length; i++) {
                            if (currMaxLocal < input->getScalar<T>(i)) {
                                currMaxLocal = input->getScalar<T>(i);
                                maxIdxLocal = i;
                            }
                            output->putScalar<T>(i, 0.f);
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
                output->putScalar<T>(maxIdx, 1.f);
            }
            else {
                int maxIdx = 0;
                T currMax = input->getScalar<T>(0);
                if (length < ELEMENT_THRESHOLD) {
//#pragma omp parallel for reduction(max:maxIdx,currMax) proc_bind(AFFINITY)
                    for (int i = 0; i < length; i++) {
                        if (currMax < input->getScalar<T>(i*eleStride)) {
                            currMax = input->getScalar<T>(i*eleStride);
                            maxIdx = i;
                        }
                        output->putScalar<T>(i, 0.f);
                    }
                }
                else {
#pragma omp parallel proc_bind(AFFINITY) default(shared)
                    {
                        int maxIdxLocal = maxIdx;
                        T currMaxLocal = currMax;
//#pragma omp parallel for reduction(max:maxIdx,currMax)  proc_bind(AFFINITY)
                        for (int i = 0; i < length; i++) {
                            if (currMaxLocal < input->getScalar<T>(i*eleStride)) {
                                currMaxLocal = input->getScalar<T>(i*eleStride);
                                       maxIdxLocal = i;
                            }
                            output->putScalar<T>(i, 0.f);
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
                output->putScalar<T>(maxIdx, 1.f);
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
        shape::TAD tad(input->getShapeInfo(), const_cast<int*>(dimensions.data()), dimensionsLength);
        tad.createTadOnlyShapeInfo();
        tad.createOffsets();

        Nd4jLong *tadShapeShapeInfo = tad.tadOnlyShapeInfo;
        Nd4jLong* tadOffsets = tad.tadOffsets;

        int tadLength = shape::tadLength(input->getShapeInfo(), const_cast<int*>(dimensions.data()), dimensionsLength);
        int tads = input->lengthOf() / tadLength;

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
                    T *rX = const_cast<NDArray*>(input)->bufferAsT<T>() + tadOffsets[r];
                    T *rZ = output->bufferAsT<T>() + tadOffsets[r];

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
                    } 
                    else {
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
                    T *xPointer = const_cast<NDArray*>(input)->bufferAsT<T>() + offset;
                    T *resultPointer = output->bufferAsT<T>() + offset;
                    T maxValue = xPointer[0];

                    T *maxCursor = resultPointer;
                    Nd4jPointer maxCursorLong = reinterpret_cast<Nd4jPointer>(maxCursor);
                    
                    if (PrepareTwoRawArrayIter<T>(rank, xShape, xPointer, xStride, resultPointer, resultStride, &rank, shapeIter, &xPointer, xStridesIter, &resultPointer, resultStridesIter) >= 0) {                    
                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); 
                        {
                            if (maxValue < xPointer[0]) {
                                maxCursor = resultPointer;
                                maxCursorLong = reinterpret_cast<Nd4jPointer>(resultPointer);
                                maxValue = xPointer[0];
                            }
                            resultPointer[0] = 0.0;
                        }
                        ND4J_RAW_ITER_TWO_NEXT(dim, rank, coord, shapeIter, xPointer, xStridesIter, resultPointer, resultStridesIter);
                        maxCursor = reinterpret_cast<T*>(maxCursorLong);
                        maxCursor[0] = 1.0;
                    }
                }
            }
        }
    }
}


void ismax(const NDArray *input, NDArray *output, const std::vector<int>& dimensions) {
    BUILD_SINGLE_SELECTOR(input->dataType(), ismax_, (input, output, dimensions), LIBND4J_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void ismax_, (const NDArray *input, NDArray *output, const std::vector<int>& dimensions), LIBND4J_TYPES);

}
}
}

