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
#include <NDArray.h>
#include <ops/declarable/CustomOperations.h>
#include <types/types.h>

namespace nd4j {

/**
* Concatneate multi array of the same shape together
* along a particular dimension
*/
template <typename T>
void SpecialMethods<T>::concatCpuGeneric(int dimension, int numArrays, Nd4jPointer *data, Nd4jPointer *inputShapeInfo, void *vresult, Nd4jLong *resultShapeInfo) {
    auto result = reinterpret_cast<T *>(vresult);

    std::vector<Nd4jLong> iArgs = {dimension};
    std::vector<T> tArgs;
    std::vector<NDArray*> inputs(numArrays);
    std::vector<NDArray*> outputs(1);

    outputs[0] = new NDArray(static_cast<void*>(result), static_cast<Nd4jLong*>(resultShapeInfo));

    for(int i = 0; i < numArrays; ++i)
        inputs[i] = new NDArray(static_cast<void *>(data[i]), static_cast<Nd4jLong*>(inputShapeInfo[i]));

    nd4j::ops::concat op;
    auto status = op.execute(inputs, outputs, tArgs, iArgs);
    if(status != Status::OK())
        throw std::runtime_error("concatCpuGeneric fails to be executed !");
    
    delete outputs[0];

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

        // aggregation step
#ifdef _OPENMP
        int _threads = omp_get_max_threads();
#else
        // we can use whatever we want here, this value won't be used if there's no omp
    int _threads = 4;
#endif

#pragma omp parallel for simd num_threads(_threads) schedule(guided) default(shared) proc_bind(close)
        for (Nd4jLong i = 0; i < length; i++) {

            for (Nd4jLong ar = 0; ar < n; ar++) {
                z[i] += x[ar][i];
            }
        }
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

#pragma omp simd
            for (Nd4jLong i = 0; i < length; i++) {
                z[i] /= n;
            }

#ifdef _OPENNMP
            int _threads = omp_get_max_threads(); //nd4j::math::nd4j_min<int>(omp_get_max_threads() / 2, 4);
#else
            // we can use whatever we want here, this value won't be used if there's no omp
            int _threads = 4;
#endif

#pragma omp parallel for simd num_threads(_threads) schedule(guided) default(shared) proc_bind(close)
            for (Nd4jLong i = 0; i < length; i++) {

                for (Nd4jLong ar = 1; ar < n; ar++) {
                    z[i] += x[ar][i] / n;
                }
            }

            // instead of doing element-wise propagation, we just issue memcpy to propagate data
#pragma omp parallel for num_threads(_threads) default(shared) proc_bind(close)
            for (Nd4jLong ar = 1; ar < n; ar++) {
                memcpy(x[ar], z, length * sizeof(T));
            }
        } else {
            // code branch for existing Z

            // memset before propagation
            memset(z, 0, length * sizeof(T));

            // aggregation step
#ifdef _OPENNMP
            int _threads = omp_get_max_threads(); //nd4j::math::nd4j_min<int>(omp_get_max_threads() / 2, 4);
#else
            // we can use whatever we want here, this value won't be used if there's no omp
            int _threads = 4;
#endif

#pragma omp parallel for simd num_threads(_threads) schedule(guided) default(shared) proc_bind(close)
            for (Nd4jLong i = 0; i < length; i++) {

                for (Nd4jLong ar = 0; ar < n; ar++) {
                    z[i] += x[ar][i] / n;
                }
            }

            // instead of doing element-wise propagation, we just issue memcpy to propagate data
#pragma omp parallel for num_threads(_threads) default(shared) proc_bind(close)
            for (Nd4jLong ar = 0; ar < n; ar++) {
                memcpy(x[ar], z, length * sizeof(T));
            }
        }
    }

    template <typename T>
    Nd4jLong SpecialMethods<T>::getPosition(Nd4jLong *xShapeInfo, Nd4jLong index) {
        auto xEWS = shape::elementWiseStride(xShapeInfo);

        if (xEWS == 1) {
            return index;
        } else if (xEWS > 1) {
            return index * xEWS;
        } else {
            Nd4jLong xCoord[MAX_RANK];
            int xRank = shape::rank(xShapeInfo);
            auto xShape = shape::shapeOf(xShapeInfo);
            auto xStride = shape::stride(xShapeInfo);

            shape::ind2subC(xRank, xShape, index, xCoord);
            auto xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);

            return xOffset;
        }
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
#pragma omp task
            { quickSort_parallel_internal(array, xShapeInfo, left, j, cutoff, descending); }
#pragma omp task
            { quickSort_parallel_internal(array, xShapeInfo, i, right, cutoff, descending); }
        }

    }

    template<typename T>
    void SpecialMethods<T>::quickSort_parallel(void *varray, Nd4jLong *xShapeInfo, Nd4jLong lenArray, int numThreads, bool descending){
        auto array = reinterpret_cast<T *>(varray);
        int cutoff = 1000;

#pragma omp parallel num_threads(numThreads)
        {
#pragma omp single nowait
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

#pragma omp parallel for
        for (int r = 0; r < numTads; r++) {
            T *dx = x + tadOffsets[r];

            quickSort_parallel(dx, tadShapeInfo, xTadLength, 1, descending);
        }
    }


    template<typename T>
    void SpecialMethods<T>::decodeBitmapGeneric(void *dx, Nd4jLong N, void *vz, Nd4jLong *zShapeInfo) {
        auto dz = reinterpret_cast<T *>(vz);
        auto x = reinterpret_cast<int *>(dx);
        Nd4jLong lim = N / 16 + 5;

        FloatBits2 fb;
        fb.i_ = x[2];
        float threshold = fb.f_;


#pragma omp parallel for schedule(guided) proc_bind(close)
        for (Nd4jLong e = 4; e < lim; e++) {

            for (int bitId = 0; bitId < 16; bitId++) {
                bool hasBit = (x[e] & 1 << (bitId) ) != 0;
                bool hasSign = (x[e] & 1 << (bitId + 16) ) != 0;

                if (hasBit) {
                    if (hasSign)
                        dz[(e - 4) * 16 + bitId] -= threshold;
                    else
                        dz[(e - 4) * 16 + bitId] += threshold;
                } else if (hasSign) {
                    dz[(e - 4) * 16 + bitId] -= threshold / 2;
                }
            }
        }
    }

    template<typename T>
    Nd4jLong SpecialMethods<T>::encodeBitmapGeneric(void *vx, Nd4jLong *xShapeInfo, Nd4jLong N, int *dz, float threshold) {
        auto dx = reinterpret_cast<T *>(vx);

        Nd4jLong retVal = 0L;

#pragma omp parallel for schedule(guided) proc_bind(close) reduction(+:retVal)
        for (Nd4jLong x = 0; x < N; x += 16) {

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
                        dx[e] += threshold;
                    } else {
                        dx[e] -= threshold;
                    }
                } else if (abs >= (T) threshold / (T) 2.0f && val < (T) 0.0f) {
                    byte |= 1 << (bitId + 16);
                    dx[e] += threshold / 2;

                    retVal++;
                }
            }

            dz[byteId] = byte;
        }

        return retVal;
    }

    BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT SpecialMethods, , LIBND4J_TYPES);
}
