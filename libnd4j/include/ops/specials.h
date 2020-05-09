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
// Created by raver119 on 24.04.17.
//

#ifndef LIBND4J_SPECIALS_H
#define LIBND4J_SPECIALS_H


#ifdef __CUDACC__
#define ELEMENT_THRESHOLD 8192
#define TAD_THRESHOLD 2
#endif

#include <system/pointercast.h>
#include <vector>

namespace sd {
    class NDArray;

    //FIXME: get rid of this redefinition
    typedef union
    {
        float f_;
        int   i_;
    } FloatBits2;


    class ND4J_EXPORT SpecialTypeConverter {
    public:
        template<typename S, typename T>
        static void convertGeneric(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz);
    };

    template <typename T>
    class ND4J_EXPORT SpecialMethods {
    public:
        static void concatCpuGeneric(const std::vector<const NDArray*>& inArrs, NDArray& output, int axis);
        static void concatCpuGeneric(int dimension, int numArrays, Nd4jPointer *data, Nd4jPointer *inputShapeInfo, void *result, Nd4jLong const* resultShapeInfo);
        static void splitCpuGeneric(const NDArray& input, const std::vector<NDArray*>& outArrs, int axis);
        static void accumulateGeneric(void **x, void *z, const Nd4jLong *zShapeInfo, int n, Nd4jLong length);
        static void averageGeneric(void **x, void *z, const Nd4jLong  *zShapeInfo, int n, Nd4jLong length, bool propagate);

        static Nd4jLong getPosition(const Nd4jLong *xShapeInfo, Nd4jLong index);
        static void quickSort_parallel_internal(T* array, const Nd4jLong *xShapeInfo, int left, int right, int cutoff, bool descending);
        static void quickSort_parallel(void* array, const Nd4jLong *xShapeInfo, Nd4jLong lenArray, int numThreads, bool descending);

        static int nextPowerOf2(int number);
        static int lastPowerOf2(int number);

        static void sortGeneric(void *x, const Nd4jLong *xShapeInfo, bool descending);
        static void sortTadGeneric(void *x, const Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets, bool descending);

        static void decodeBitmapGeneric(const void *dx, Nd4jLong N, void *dz, const Nd4jLong *zShapeInfo);
        static Nd4jLong encodeBitmapGeneric(void *dx, const Nd4jLong *zShapeInfo, Nd4jLong N, int *dz, float threshold);

    };

    template <typename X, typename Y>
    class ND4J_EXPORT DoubleMethods{
    public:
        static void sortByKey(void *vx, Nd4jLong const* xShapeInfo, void *vy, Nd4jLong const* yShapeInfo, bool descending);
        static void sortByValue(void *vx, Nd4jLong const* xShapeInfo, void *vy, Nd4jLong const* yShapeInfo, bool descending);


        static void sortTadByKey(void *vx, Nd4jLong const* xShapeInfo, void *vy, Nd4jLong const* yShapeInfo, int *dimension, int dimensionLength, bool descending);
        static void sortTadByValue(void *vx, Nd4jLong const* xShapeInfo, void *vy, Nd4jLong const* yShapeInfo, int *dimension, int dimensionLength, bool descending);
    };
}


#endif //LIBND4J_SPECIALS_H
