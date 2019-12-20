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


#include <pointercast.h>
#include <helpers/shape.h>
#include <helpers/TAD.h>
#include <specials.h>
#include <dll.h>
#include <NDArray.h>
#include <ops/declarable/CustomOperations.h>
#include <types/types.h>
#include <helpers/Loops.h>

namespace nd4j {

/**
* Concatneate multi array of the same shape together
* along a particular dimension
*/
template <typename T>
void SpecialMethods<T>::concatCpuGeneric(const std::vector<NDArray*>& inArrs, NDArray& output, const int axis) {
        const uint numOfArrs = inArrs.size();

        int outDim;
        const bool isOutputVector = output.isCommonVector(outDim);

        if(isOutputVector || (axis == 0 && output.ordering() == 'c')) {

            bool allVectorsOrScalars = true;
            const uint outEws = isOutputVector ? output.stridesOf()[outDim] : output.ews();

            std::vector<int> nonUnityDim(numOfArrs);
            std::vector<Nd4jLong> zOffset(numOfArrs);

            for(int i = 0; i < numOfArrs; i++) {
                allVectorsOrScalars &= (inArrs[i]->lengthOf() == 1 || inArrs[i]->isCommonVector(nonUnityDim[i]));
                if(!allVectorsOrScalars)
                    break;
                if(i == 0)  zOffset[0] = 0;
                else        zOffset[i] = zOffset[i - 1] + outEws * inArrs[i - 1]->lengthOf();
            }

            if(allVectorsOrScalars) {

                T* outBuff = output.bufferAsT<T>();

                auto func = PRAGMA_THREADS_FOR {
                    for (auto r = start; r < stop; r += increment) {
                        const Nd4jLong arrLen = inArrs[r]->lengthOf();
                        const uint xEws = (arrLen == 1) ? 1 : inArrs[r]->stridesOf()[nonUnityDim[r]];

                        T *z = outBuff + zOffset[r];
                        T *x = inArrs[r]->bufferAsT<T>();

                        if (outEws == 1 && xEws == 1)
                            for (Nd4jLong e = 0; e < arrLen; e++)
                                z[e] = x[e];
                        else
                            for (Nd4jLong e = 0; e < arrLen; e++)
                                z[e * outEws] = x[e * xEws];
                    }
                };

                samediff::Threads::parallel_tad(func, 0, numOfArrs);
                return;
            }
        }

        const int rank  = inArrs[0]->rankOf();
        const int rank2 = 2*rank;
        std::vector<std::vector<Nd4jLong>> indices(numOfArrs, std::vector<Nd4jLong>(rank2,0));

        // take into account indices for first array
        indices[0][2 * axis + 1] = inArrs[0]->sizeAt(axis);

        // loop through the rest of input arrays
        for(int i = 1; i < numOfArrs; ++i) {
            indices[i][2 * axis]     = indices[i-1][2 * axis + 1];                                // index start from
            indices[i][2 * axis + 1] = indices[i-1][2 * axis + 1] + inArrs[i]->sizeAt(axis);      // index end with (excluding)
        }

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i += increment) {
                auto temp = output(indices[i], true);
                nd4j::TransformLoops<T, T, T>::template loopTransform<simdOps::Assign<T, T>>( inArrs[i]->bufferAsT<T>(), inArrs[i]->getShapeInfo(), temp.bufferAsT<T>(), temp.getShapeInfo(), nullptr, 0, 1);
            }
        };

        samediff::Threads::parallel_tad(func, 0, numOfArrs);
}

/**
* Concatneate multi array of the same shape together
* along a particular dimension
*/
template <typename T>
void SpecialMethods<T>::concatCpuGeneric(int dimension, int numArrays, Nd4jPointer *data, Nd4jPointer *inputShapeInfo, void *vresult, Nd4jLong *resultShapeInfo) {
    auto result = reinterpret_cast<T *>(vresult);
    std::vector<NDArray*> inputs(numArrays);

    NDArray output(static_cast<void*>(result), static_cast<Nd4jLong*>(resultShapeInfo));

    for(int i = 0; i < numArrays; ++i)
        inputs[i] = new NDArray(static_cast<void *>(data[i]), static_cast<Nd4jLong*>(inputShapeInfo[i]));

    nd4j::SpecialMethods<T>::concatCpuGeneric(inputs, output, dimension);

    for(int i = 0; i < numArrays; ++i)
        delete inputs[i];
}

/**
 * This kernel accumulates X arrays, and stores result into Z
 *
 * @tparam T
 * @param x
 * @param z
 * @param n
 * @param length
 */
    template<typename T>
    void SpecialMethods<T>::accumulateGeneric(void **vx, void *vz, Nd4jLong *zShapeInfo, int n, const Nd4jLong length) {
        auto z = reinterpret_cast<T *>(vz);
        auto x = reinterpret_cast<T **>(vx);

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i += increment) {
                for (auto ar = 0L; ar < n; ar++) {
                    z[i] += x[ar][i];
                }
            }
        };

        samediff::Threads::parallel_for(func, 0, length);
    }


/**
 * This kernel averages X input arrays, and stores result to Z
 *
 * @tparam T
 * @param x
 * @param z
 * @param n
 * @param length
 * @param propagate
 */
    template<typename T>
    void SpecialMethods<T>::averageGeneric(void **vx, void *vz, Nd4jLong *zShapeInfo, int n, const Nd4jLong length, bool propagate) {
        auto z = reinterpret_cast<T *>(vz);
        auto x = reinterpret_cast<T **>(vx);

        if (z == nullptr) {
            //code branch for absent Z
            z = x[0];

            PRAGMA_OMP_SIMD
            for (uint64_t i = 0; i < length; i++) {
                z[i] /= static_cast<T>(n);
            }

            auto func = PRAGMA_THREADS_FOR {
                for (auto i = start; i < stop; i += increment) {
                    for (Nd4jLong ar = 1; ar < n; ar++) {
                        z[i] += x[ar][i] / static_cast<T>(n);
                    }
                }
            };
            samediff::Threads::parallel_for(func, 0, length);

            // instead of doing element-wise propagation, we just issue memcpy to propagate data
            for (Nd4jLong ar = 1; ar < n; ar++) {
                memcpy(x[ar], z, length * sizeof(T));
            }
        } else {
            // code branch for existing Z

            // memset before propagation
            memset(z, 0, length * sizeof(T));

            // aggregation step
            auto func = PRAGMA_THREADS_FOR {
                for (auto i = start; i < stop; i += increment) {
                    for (Nd4jLong ar = 0; ar < n; ar++) {
                        z[i] += x[ar][i] / static_cast<T>(n);
                    }
                }
            };
            samediff::Threads::parallel_for(func, 0, length);

            // instead of doing element-wise propagation, we just issue memcpy to propagate data
            for (Nd4jLong ar = 0; ar < n; ar++) {
                memcpy(x[ar], z, length * sizeof(T));
            }
        }
    }

    template <typename T>
    Nd4jLong SpecialMethods<T>::getPosition(Nd4jLong *xShapeInfo, Nd4jLong index) {
        auto xEWS = shape::elementWiseStride(xShapeInfo);

        if (xEWS == 1)
            return index;
        else if (xEWS > 1)
            return index * xEWS;
        else
            return shape::getIndexOffset(index, xShapeInfo);
    }

    template<typename T>
    void SpecialMethods<T>::quickSort_parallel_internal(T* array, Nd4jLong *xShapeInfo, int left, int right, int cutoff, bool descending) {

        int i = left, j = right;
        T tmp;
        T pivot = array[getPosition(xShapeInfo, (left + right) / 2)];


        {
            /* PARTITION PART */
            while (i <= j) {
                if (descending) {
                    while (array[getPosition(xShapeInfo, i)] > pivot)
                        i++;
                    while (array[getPosition(xShapeInfo, j)] < pivot)
                        j--;
                    if (i <= j) {
                        tmp = array[getPosition(xShapeInfo, i)];
                        array[getPosition(xShapeInfo, i)] = array[getPosition(xShapeInfo, j)];
                        array[getPosition(xShapeInfo, j)] = tmp;
                        i++;
                        j--;
                    }
                } else {
                    while (array[getPosition(xShapeInfo, i)] < pivot)
                        i++;
                    while (array[getPosition(xShapeInfo, j)] > pivot)
                        j--;
                    if (i <= j) {
                        tmp = array[getPosition(xShapeInfo, i)];
                        array[getPosition(xShapeInfo, i)] = array[getPosition(xShapeInfo, j)];
                        array[getPosition(xShapeInfo, j)] = tmp;
                        i++;
                        j--;
                    }
                }
            }

        }

        //

        if ( ((right-left)<cutoff) ){
            if (left < j){ quickSort_parallel_internal(array, xShapeInfo, left, j, cutoff, descending); }
            if (i < right){ quickSort_parallel_internal(array, xShapeInfo, i, right, cutoff, descending); }

        }else{
PRAGMA_OMP_TASK
            { quickSort_parallel_internal(array, xShapeInfo, left, j, cutoff, descending); }
PRAGMA_OMP_TASK
            { quickSort_parallel_internal(array, xShapeInfo, i, right, cutoff, descending); }
        }
    }

    template<typename T>
    void SpecialMethods<T>::quickSort_parallel(void *varray, Nd4jLong *xShapeInfo, Nd4jLong lenArray, int numThreads, bool descending){
        auto array = reinterpret_cast<T *>(varray);
        int cutoff = 1000;

        PRAGMA_OMP_PARALLEL_THREADS(numThreads)
        {
PRAGMA_OMP_SINGLE_ARGS(nowait)
            {
                quickSort_parallel_internal(array, xShapeInfo, 0, lenArray-1, cutoff, descending);
            }
        }

    }



    template <typename T>
    int SpecialMethods<T>::nextPowerOf2(int number) {
        int pos = 0;

        while (number > 0) {
            pos++;
            number = number >> 1;
        }
        return (int) pow(2, pos);
    }

    template <typename T>
    int SpecialMethods<T>::lastPowerOf2(int number) {
        int p = 1;
        while (p <= number)
            p <<= 1;

        p >>= 1;
        return p;
    }


    template<typename T>
    void SpecialMethods<T>::sortGeneric(void *vx, Nd4jLong *xShapeInfo, bool descending) {
        auto x = reinterpret_cast<T *>(vx);

        quickSort_parallel(x, xShapeInfo, shape::length(xShapeInfo), omp_get_max_threads(), descending);
    }

    template<typename T>
    void SpecialMethods<T>::sortTadGeneric(void *vx, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending) {
        auto x = reinterpret_cast<T *>(vx);

        //quickSort_parallel(x, xShapeInfo, shape::length(xShapeInfo), omp_get_max_threads(), descending);
        Nd4jLong xLength = shape::length(xShapeInfo);
        Nd4jLong xTadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
        int numTads = xLength / xTadLength;

        auto func = PRAGMA_THREADS_FOR {
            for (auto r = start; r < stop; r += increment) {
                T *dx = x + tadOffsets[r];

                quickSort_parallel(dx, tadShapeInfo, xTadLength, 1, descending);
            }
        };
        samediff::Threads::parallel_tad(func, 0, numTads);
    }


    template<typename T>
    void SpecialMethods<T>::decodeBitmapGeneric(void *dx, Nd4jLong N, void *vz, Nd4jLong *zShapeInfo) {
        auto dz = reinterpret_cast<T *>(vz);
        auto x = reinterpret_cast<int *>(dx);
        Nd4jLong lim = N / 16 + 5;

        FloatBits2 fb;
        fb.i_ = x[2];
        float threshold = fb.f_;


        auto func = PRAGMA_THREADS_FOR {
            for (auto e = start; e < stop; e += increment) {
                for (int bitId = 0; bitId < 16; bitId++) {
                    bool hasBit = (x[e] & 1 << (bitId)) != 0;
                    bool hasSign = (x[e] & 1 << (bitId + 16)) != 0;

                    if (hasBit) {
                        if (hasSign)
                            dz[(e - 4) * 16 + bitId] -= static_cast<T>(threshold);
                        else
                            dz[(e - 4) * 16 + bitId] += static_cast<T>(threshold);
                    } else if (hasSign) {
                        dz[(e - 4) * 16 + bitId] -= static_cast<T>(threshold / 2);
                    }
                }
            }
        };

        samediff::Threads::parallel_for(func, 4, lim);
    }

    template<typename S, typename T>
    void SpecialTypeConverter::convertGeneric(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz) {
        auto x = reinterpret_cast<S *>(dx);
        auto z = reinterpret_cast<T *>(dz);


        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i += increment) {
                z[i] = static_cast<T>(x[i]);
            }
        };

        samediff::Threads::parallel_for(func, 0, N);
    };
    BUILD_DOUBLE_TEMPLATE(template void SpecialTypeConverter::convertGeneric, (Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz), LIBND4J_TYPES, LIBND4J_TYPES);

    template<typename T>
    Nd4jLong SpecialMethods<T>::encodeBitmapGeneric(void *vx, Nd4jLong *xShapeInfo, Nd4jLong N, int *dz, float threshold) {
        auto dx = reinterpret_cast<T *>(vx);

//PRAGMA_OMP_PARALLEL_FOR_ARGS(schedule(guided) proc_bind(close) reduction(+:retVal))
        auto func = PRAGMA_REDUCE_LONG {
            Nd4jLong retVal = 0L;

            for (auto x = start; x < stop; x += increment) {
                int byte = 0;
                int byteId = x / 16 + 4;

                for (int f = 0; f < 16; f++) {
                    Nd4jLong e = x + f;

                    if (e >= N)
                        continue;

                    T val = dx[e];
                    T abs = nd4j::math::nd4j_abs<T>(val);

                    int bitId = e % 16;

                    if (abs >= (T) threshold) {
                        byte |= 1 << (bitId);
                        retVal++;

                        if (val < (T) 0.0f) {
                            byte |= 1 << (bitId + 16);
                            dx[e] += static_cast<T>(threshold);
                        } else {
                            dx[e] -= static_cast<T>(threshold);
                        }
                    } else if (abs >= (T) threshold / (T) 2.0f && val < (T) 0.0f) {
                        byte |= 1 << (bitId + 16);
                        dx[e] += static_cast<T>(threshold / 2);

                        retVal++;
                    }
                }

                dz[byteId] = byte;
            }

            return retVal;
        };
        return samediff::Threads::parallel_long(func, LAMBDA_SUML, 0, N, 16);
    }

    template <typename X, typename Y>
    void quickSort_parallel_internal_key(X* key, Nd4jLong *xShapeInfo, Y* values, Nd4jLong *yShapeInfo, int left, int right, int cutoff, bool descending) {
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
    void quickSort_parallel_internal_value(X* key, Nd4jLong *xShapeInfo, Y* value, Nd4jLong *yShapeInfo, int left, int right, int cutoff, bool descending) {
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
    static void quickSort_parallel_key(void *varray, Nd4jLong *xShapeInfo, void *yarray, Nd4jLong *yShapeInfo, Nd4jLong lenArray, int numThreads, bool descending){
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
    static void quickSort_parallel_value(void *varray, Nd4jLong *xShapeInfo, void *yarray, Nd4jLong *yShapeInfo, Nd4jLong lenArray, int numThreads, bool descending){
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
    void DoubleMethods<X,Y>::sortByKey(void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, bool descending) {
        quickSort_parallel_key<X,Y>(vx, xShapeInfo, vy, yShapeInfo, shape::length(xShapeInfo), omp_get_max_threads(), descending);
    }

    template <typename X, typename Y>
    void DoubleMethods<X,Y>::sortByValue(void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, bool descending) {
        quickSort_parallel_value<X,Y>(vx, xShapeInfo, vy, yShapeInfo, shape::length(xShapeInfo), omp_get_max_threads(), descending);
    }

    template <typename X, typename Y>
    void DoubleMethods<X,Y>::sortTadByKey(void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, int *dimension, int dimensionLength, bool descending) {
        auto x = reinterpret_cast<X*>(vx);
        auto y = reinterpret_cast<Y*>(vy);

        auto packX = ConstantTadHelper::getInstance()->tadForDimensions(xShapeInfo, dimension, dimensionLength);
        auto packY = ConstantTadHelper::getInstance()->tadForDimensions(yShapeInfo, dimension, dimensionLength);

        auto xLength = shape::length(xShapeInfo);
        auto xTadLength = shape::length(packX.primaryShapeInfo());
        auto numTads = packX.numberOfTads();

        auto func = PRAGMA_THREADS_FOR {
            for (auto r = start; r < stop; r += increment) {
                auto dx = x + packX.primaryOffsets()[r];
                auto dy = y + packY.primaryOffsets()[r];

                quickSort_parallel_key<X, Y>(dx, packX.primaryShapeInfo(), dy, packY.primaryShapeInfo(), xTadLength, 1, descending);
            }
        };

        samediff::Threads::parallel_tad(func, 0, numTads);
    }

    template <typename X, typename Y>
    void DoubleMethods<X,Y>::sortTadByValue(void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, int *dimension, int dimensionLength, bool descending) {
        auto x = reinterpret_cast<X*>(vx);
        auto y = reinterpret_cast<Y*>(vy);

        auto packX = ConstantTadHelper::getInstance()->tadForDimensions(xShapeInfo, dimension, dimensionLength);
        auto packY = ConstantTadHelper::getInstance()->tadForDimensions(yShapeInfo, dimension, dimensionLength);

        auto xLength = shape::length(xShapeInfo);
        auto xTadLength = shape::length(packX.primaryShapeInfo());
        auto numTads = packX.numberOfTads();

        auto func = PRAGMA_THREADS_FOR {
            for (auto r = start; r < stop; r += increment) {
                auto dx = x + packX.primaryOffsets()[r];
                auto dy = y + packY.primaryOffsets()[r];

                quickSort_parallel_value<X, Y>(dx, packX.primaryShapeInfo(), dy, packY.primaryShapeInfo(), xTadLength, 1, descending);
            }
        };

        samediff::Threads::parallel_tad(func, 0, numTads);
    }

    BUILD_SINGLE_TEMPLATE(template class SpecialMethods, , LIBND4J_TYPES);
    BUILD_DOUBLE_TEMPLATE(template class DoubleMethods, , LIBND4J_TYPES, LIBND4J_TYPES);
}

