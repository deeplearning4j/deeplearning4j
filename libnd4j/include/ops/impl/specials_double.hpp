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
// @author raver119@gmail.com, created on 07.10.2017.
// @author Yurii Shyrma (iuriish@yahoo.com)
//


#include <system/pointercast.h>
#include <helpers/shape.h>
#include <helpers/TAD.h>
#include <ops/specials.h>
#include <system/dll.h>
#include <array/NDArray.h>
#include <ops/declarable/CustomOperations.h>
#include <types/types.h>
#include <helpers/Loops.h>

namespace sd {


    template<typename S, typename T>
    void SpecialTypeConverter::convertGeneric(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz) {
        auto x = reinterpret_cast<S *>(dx);
        auto z = reinterpret_cast<T *>(dz);


        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
                z[i] = static_cast<T>(x[i]);
            }
        };

        samediff::Threads::parallel_for(func, 0, N);
    };


    template <typename X, typename Y>
    void quickSort_parallel_internal_key(X* key, Nd4jLong const* xShapeInfo, Y* values, Nd4jLong const* yShapeInfo, int left, int right, int cutoff, bool descending) {
        int i = left, j = right;
        X ktmp;
        X pivot = key[shape::getIndexOffset((left + right) / 2, xShapeInfo)];

        Y vtmp;

        {
            /* PARTITION PART */
            while (i <= j) {
                if (descending) {
                    while (key[shape::getIndexOffset(i, xShapeInfo)] > pivot)
                        i++;
                    while (key[shape::getIndexOffset(j, xShapeInfo)] < pivot)
                        j--;
                    if (i <= j) {
                        ktmp = key[shape::getIndexOffset(i, xShapeInfo)];
                        key[shape::getIndexOffset(i, xShapeInfo)] = key[shape::getIndexOffset(j, xShapeInfo)];
                        key[shape::getIndexOffset(j, xShapeInfo)] = ktmp;

                        vtmp = values[shape::getIndexOffset(i, yShapeInfo)];
                        values[shape::getIndexOffset(i, yShapeInfo)] = values[shape::getIndexOffset(j, yShapeInfo)];
                        values[shape::getIndexOffset(j, yShapeInfo)] = vtmp;

                        i++;
                        j--;
                    }
                } else {
                    while (key[shape::getIndexOffset(i, xShapeInfo)] < pivot)
                        i++;
                    while (key[shape::getIndexOffset(j, xShapeInfo)] > pivot)
                        j--;
                    if (i <= j) {
                        ktmp = key[shape::getIndexOffset(i, xShapeInfo)];
                        key[shape::getIndexOffset(i, xShapeInfo)] = key[shape::getIndexOffset(j, xShapeInfo)];
                        key[shape::getIndexOffset(j, xShapeInfo)] = ktmp;

                        vtmp = values[shape::getIndexOffset(i, yShapeInfo)];
                        values[shape::getIndexOffset(i, yShapeInfo)] = values[shape::getIndexOffset(j, yShapeInfo)];
                        values[shape::getIndexOffset(j, yShapeInfo)] = vtmp;

                        i++;
                        j--;
                    }
                }
            }

        }

        //

        if ( ((right-left)<cutoff) ){
            if (left < j){ quickSort_parallel_internal_key(key, xShapeInfo, values, yShapeInfo, left, j, cutoff, descending); }
            if (i < right){ quickSort_parallel_internal_key(key, xShapeInfo, values, yShapeInfo, i, right, cutoff, descending); }

        }else{
PRAGMA_OMP_TASK
            { quickSort_parallel_internal_key(key, xShapeInfo, values, yShapeInfo, left, j, cutoff, descending); }
PRAGMA_OMP_TASK
            { quickSort_parallel_internal_key(key, xShapeInfo, values, yShapeInfo, i, right, cutoff, descending); }
        }
    }


    template <typename X, typename Y>
    void quickSort_parallel_internal_value(X* key, Nd4jLong const* xShapeInfo, Y* value, Nd4jLong const* yShapeInfo, int left, int right, int cutoff, bool descending) {
        int i = left, j = right;
        X ktmp;
        Y pivot = value[shape::getIndexOffset((left + right) / 2, yShapeInfo)];

        Y vtmp;

        {
            /* PARTITION PART */
            while (i <= j) {
                if (descending) {
                    while (value[shape::getIndexOffset(i, yShapeInfo)] > pivot)
                        i++;
                    while (value[shape::getIndexOffset(j, yShapeInfo)] < pivot)
                        j--;
                    if (i <= j) {
                        ktmp = key[shape::getIndexOffset(i, xShapeInfo)];
                        key[shape::getIndexOffset(i, xShapeInfo)] = key[shape::getIndexOffset(j, xShapeInfo)];
                        key[shape::getIndexOffset(j, xShapeInfo)] = ktmp;

                        vtmp = value[shape::getIndexOffset(i, yShapeInfo)];
                        value[shape::getIndexOffset(i, yShapeInfo)] = value[shape::getIndexOffset(j, yShapeInfo)];
                        value[shape::getIndexOffset(j, yShapeInfo)] = vtmp;

                        i++;
                        j--;
                    }
                } else {
                    while (value[shape::getIndexOffset(i, yShapeInfo)] < pivot)
                        i++;
                    while (value[shape::getIndexOffset(j, yShapeInfo)] > pivot)
                        j--;
                    if (i <= j) {
                        ktmp = key[shape::getIndexOffset(i, xShapeInfo)];
                        key[shape::getIndexOffset(i, xShapeInfo)] = key[shape::getIndexOffset(j, xShapeInfo)];
                        key[shape::getIndexOffset(j, xShapeInfo)] = ktmp;

                        vtmp = value[shape::getIndexOffset(i, yShapeInfo)];
                        value[shape::getIndexOffset(i, yShapeInfo)] = value[shape::getIndexOffset(j, yShapeInfo)];
                        value[shape::getIndexOffset(j, yShapeInfo)] = vtmp;

                        i++;
                        j--;
                    }
                }
            }

        }

        //

        if ( ((right-left)<cutoff) ){
            if (left < j){ quickSort_parallel_internal_value(key, xShapeInfo, value, yShapeInfo, left, j, cutoff, descending); }
            if (i < right){ quickSort_parallel_internal_value(key, xShapeInfo, value, yShapeInfo, i, right, cutoff, descending); }

        }else{
PRAGMA_OMP_TASK
            { quickSort_parallel_internal_value(key, xShapeInfo, value, yShapeInfo, left, j, cutoff, descending); }
PRAGMA_OMP_TASK
            { quickSort_parallel_internal_value(key, xShapeInfo, value, yShapeInfo, i, right, cutoff, descending); }
        }
    }


    template <typename X, typename Y>
    static void quickSort_parallel_key(void *varray, Nd4jLong const* xShapeInfo, void *yarray, Nd4jLong const* yShapeInfo, Nd4jLong lenArray, int numThreads, bool descending){
        auto array = reinterpret_cast<X *>(varray);
        auto values = reinterpret_cast<Y *>(yarray);
        int cutoff = 1000;

        PRAGMA_OMP_PARALLEL_THREADS(numThreads)
        {
PRAGMA_OMP_SINGLE_ARGS(nowait)
            {
                quickSort_parallel_internal_key(array, xShapeInfo, values, yShapeInfo, 0, lenArray-1, cutoff, descending);
            }
        }
    }

    template <typename X, typename Y>
    static void quickSort_parallel_value(void *varray, Nd4jLong const* xShapeInfo, void *yarray, Nd4jLong const* yShapeInfo, Nd4jLong lenArray, int numThreads, bool descending){
        auto array = reinterpret_cast<X *>(varray);
        auto values = reinterpret_cast<Y *>(yarray);
        int cutoff = 1000;

        PRAGMA_OMP_PARALLEL_THREADS(numThreads)
        {
PRAGMA_OMP_SINGLE_ARGS(nowait)
            {
                quickSort_parallel_internal_value(array, xShapeInfo, values, yShapeInfo, 0, lenArray-1, cutoff, descending);
            }
        }
    }

    template <typename X, typename Y>
    void DoubleMethods<X,Y>::sortByKey(void *vx, Nd4jLong const* xShapeInfo, void *vy, Nd4jLong const* yShapeInfo, bool descending) {
        quickSort_parallel_key<X,Y>(vx, xShapeInfo, vy, yShapeInfo, shape::length(xShapeInfo), omp_get_max_threads(), descending);
    }

    template <typename X, typename Y>
    void DoubleMethods<X,Y>::sortByValue(void *vx, Nd4jLong const* xShapeInfo, void *vy, Nd4jLong const* yShapeInfo, bool descending) {
        quickSort_parallel_value<X,Y>(vx, xShapeInfo, vy, yShapeInfo, shape::length(xShapeInfo), omp_get_max_threads(), descending);
    }

    template <typename X, typename Y>
    void DoubleMethods<X,Y>::sortTadByKey(void *vx, Nd4jLong const* xShapeInfo, void *vy, Nd4jLong const* yShapeInfo, int *dimension, int dimensionLength, bool descending) {
        auto x = reinterpret_cast<X*>(vx);
        auto y = reinterpret_cast<Y*>(vy);

        auto packX = ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimension, dimensionLength);
        auto packY = ConstantTadHelper::getInstance().tadForDimensions(yShapeInfo, dimension, dimensionLength);

        auto xLength = shape::length(xShapeInfo);
        auto xTadLength = shape::length(packX.primaryShapeInfo());
        auto numTads = packX.numberOfTads();

        auto func = PRAGMA_THREADS_FOR {
            for (auto r = start; r < stop; r++) {
                auto dx = x + packX.primaryOffsets()[r];
                auto dy = y + packY.primaryOffsets()[r];

                quickSort_parallel_key<X, Y>(dx, packX.primaryShapeInfo(), dy, packY.primaryShapeInfo(), xTadLength, 1, descending);
            }
        };

        samediff::Threads::parallel_tad(func, 0, numTads);
    }

    template <typename X, typename Y>
    void DoubleMethods<X,Y>::sortTadByValue(void *vx, Nd4jLong const* xShapeInfo, void *vy, Nd4jLong const* yShapeInfo, int *dimension, int dimensionLength, bool descending) {
        auto x = reinterpret_cast<X*>(vx);
        auto y = reinterpret_cast<Y*>(vy);

        auto packX = ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimension, dimensionLength);
        auto packY = ConstantTadHelper::getInstance().tadForDimensions(yShapeInfo, dimension, dimensionLength);

        auto xLength = shape::length(xShapeInfo);
        auto xTadLength = shape::length(packX.primaryShapeInfo());
        auto numTads = packX.numberOfTads();

        auto func = PRAGMA_THREADS_FOR {
            for (auto r = start; r < stop; r++) {
                auto dx = x + packX.primaryOffsets()[r];
                auto dy = y + packY.primaryOffsets()[r];

                quickSort_parallel_value<X, Y>(dx, packX.primaryShapeInfo(), dy, packY.primaryShapeInfo(), xTadLength, 1, descending);
            }
        };

        samediff::Threads::parallel_tad(func, 0, numTads);
    }
}

