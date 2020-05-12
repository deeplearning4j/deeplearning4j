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
/**
* Concatneate multi array of the same shape together
* along a particular dimension
*/
// template <typename T>
// void SpecialMethods<T>::concatCpuGeneric(const std::vector<NDArray*>& inArrs, NDArray& output, const int axis) {
//         const uint numOfArrs = inArrs.size();

//         int outDim;
//         const bool isOutputVector = output.isCommonVector(outDim);

//         if(isOutputVector || (axis == 0 && output.ordering() == 'c')) {

//             bool allVectorsOrScalars = true;
//             const uint outEws = isOutputVector ? output.stridesOf()[outDim] : output.ews();

//             std::vector<int> nonUnityDim(numOfArrs);
//             std::vector<Nd4jLong> zOffset(numOfArrs);

//             for(int i = 0; i < numOfArrs; i++) {
//                 allVectorsOrScalars &= (inArrs[i]->lengthOf() == 1 || inArrs[i]->isCommonVector(nonUnityDim[i]));
//                 if(!allVectorsOrScalars)
//                     break;
//                 if(i == 0)  zOffset[0] = 0;
//                 else        zOffset[i] = zOffset[i - 1] + outEws * inArrs[i - 1]->lengthOf();
//             }

//             if(allVectorsOrScalars) {

//                 T* outBuff = output.bufferAsT<T>();

//                 auto func = PRAGMA_THREADS_FOR {
//                     for (auto r = start; r < stop; r += increment) {
//                         const Nd4jLong arrLen = inArrs[r]->lengthOf();
//                         const uint xEws = (arrLen == 1) ? 1 : inArrs[r]->stridesOf()[nonUnityDim[r]];

//                         T *z = outBuff + zOffset[r];
//                         T *x = inArrs[r]->bufferAsT<T>();

//                         if (outEws == 1 && xEws == 1)
//                             for (Nd4jLong e = 0; e < arrLen; e++)
//                                 z[e] = x[e];
//                         else
//                             for (Nd4jLong e = 0; e < arrLen; e++)
//                                 z[e * outEws] = x[e * xEws];
//                     }
//                 };

//                 samediff::Threads::parallel_tad(func, 0, numOfArrs);
//                 return;
//             }
//         }

//         const int rank  = inArrs[0]->rankOf();
//         const int rank2 = 2*rank;
//         std::vector<std::vector<Nd4jLong>> indices(numOfArrs, std::vector<Nd4jLong>(rank2,0));

//         // take into account indices for first array
//         indices[0][2 * axis + 1] = inArrs[0]->sizeAt(axis);

//         // loop through the rest of input arrays
//         for(int i = 1; i < numOfArrs; ++i) {
//             indices[i][2 * axis]     = indices[i-1][2 * axis + 1];                                // index start from
//             indices[i][2 * axis + 1] = indices[i-1][2 * axis + 1] + inArrs[i]->sizeAt(axis);      // index end with (excluding)
//         }

//         auto func = PRAGMA_THREADS_FOR {
//             for (auto i = start; i < stop; i += increment) {
//                 auto temp = output(indices[i], true);
//                 sd::TransformLoops<T, T, T>::template loopTransform<simdOps::Assign<T, T>>( inArrs[i]->bufferAsT<T>(), inArrs[i]->shapeInfo(), temp.bufferAsT<T>(), temp.shapeInfo(), nullptr, 0, 1);
//             }
//         };

//         samediff::Threads::parallel_tad(func, 0, numOfArrs);
// }

// static Nd4jLong strideOverContigAxis(const int axis, const Nd4jLong* inShapeInfo) {

//     Nd4jLong result = 9223372036854775807LL;

//     for(uint i = 0; i < shape::rank(inShapeInfo); ++i) {

//         const auto currentStride = shape::stride(inShapeInfo)[i];

//         if(i == axis || shape::shapeOf(inShapeInfo)[i] == 1)
//             continue;

//         if(result > currentStride)
//             result = currentStride;
//     }

//     return result == 9223372036854775807LL ? 1 : result;
// }


template <typename T>
void SpecialMethods<T>::concatCpuGeneric(const std::vector<const NDArray*>& inArrs, NDArray& output, const int axis) {

    const int numOfInArrs = inArrs.size();
    const auto sizeofT    = output.sizeOfT();

    T* zBuff = output.bufferAsT<T>();

    bool luckCase1 = ((axis == 0 && output.ordering() == 'c') || (axis == output.rankOf() - 1 && output.ordering() == 'f')) && output.ews() == 1;

    if(luckCase1) {
        for (uint i = 0; i < numOfInArrs; ++i) {
            luckCase1 &= inArrs[i]->ordering() == output.ordering() && inArrs[i]->ews() == 1;
            if(!luckCase1)
                break;
        }
    }

    if(luckCase1) {     // for example {1,10} + {2,10} + {3,10} = {6, 10} order c; or {10,1} + {10,2} + {10,3} = {10, 6} order f

        T* z = zBuff;
        for (uint i = 0; i < numOfInArrs; ++i) {
            const auto memAmountToCopy = inArrs[i]->lengthOf();
            memcpy(z, inArrs[i]->bufferAsT<T>(), memAmountToCopy * sizeofT);
            z += memAmountToCopy;
        }
        return;
    }

    // const bool isZcontin = output.strideAt(axis) == 1;
    // bool areInputsContin = true;
    // bool allSameOrder    = true;
    // std::vector<Nd4jLong> strideOfContigStride(numOfInArrs);

    // if(isZcontin) {

    //     for (uint i = 0; i < numOfInArrs; ++i) {

    //         areInputsContin &= inArrs[i]->strideAt(axis) == 1;
    //         allSameOrder    &= inArrs[i]->ordering() == output.ordering();
    //         if(!areInputsContin || !allSameOrder)
    //             break;

    //         strideOfContigStride[i] = strideOverContigAxis(axis, inArrs[i]->getShapeInfo());
    //     }
    // }

    // const bool luckCase2 = isZcontin && areInputsContin && allSameOrder;

    // if(luckCase2) {     // for example {2,1,3} + {2,5,3} + {2,10,3} = {2,16,3}, here axis 1 shoud have stride = 1 for all inputs arrays and output array

    //     const auto zStep = strideOverContigAxis(axis, output.getShapeInfo());

    //     for (uint i = 0; i < output.lengthOf() / output.sizeAt(axis); ++i) {

    //         T* z = zBuff + zStep * i;

    //         for (uint j = 0; j < inArrs.size(); ++j) {
    //             const auto xDim = inArrs[j]->sizeAt(axis);
    //             const T* x = inArrs[j]->bufferAsT<T>() + strideOfContigStride[j] * i;
    //             memcpy(z, x, xDim * sizeofT);
    //             z += xDim;
    //         }
    //     }

    //     return;
    // }

    // general case
    auto func = PRAGMA_THREADS_FOR {

        int coords[MAX_RANK], temp;

        for (auto i = start; i < stop; i += increment) {

            shape::index2coordsCPU(start, i, output.shapeInfo(), coords);

            const auto zOffset = shape::getOffset(output.shapeInfo(), coords);

            uint inArrIdx = 0;
            uint xDim = inArrs[inArrIdx]->sizeAt(axis);

            temp = coords[axis];
            while (coords[axis] >= xDim) {
                coords[axis] -= xDim;
                xDim = inArrs[++inArrIdx]->sizeAt(axis);
            }

            const T* x = inArrs[inArrIdx]->bufferAsT<T>();
            const auto xOffset = shape::getOffset(inArrs[inArrIdx]->shapeInfo(), coords);

            zBuff[zOffset] = x[xOffset];

            coords[axis] = temp;
        }
    };

    samediff::Threads::parallel_for(func, 0, output.lengthOf());
}

/**
* Concatneate multi array of the same shape together
* along a particular dimension
*/
template <typename T>
void SpecialMethods<T>::concatCpuGeneric(int dimension, int numArrays, Nd4jPointer *data, Nd4jPointer *inputShapeInfo, void *vresult, Nd4jLong const* resultShapeInfo) {
    auto result = reinterpret_cast<T *>(vresult);
    std::vector<const NDArray*> inputs(numArrays);

    NDArray output(static_cast<void*>(result), resultShapeInfo);

    for(int i = 0; i < numArrays; ++i)
        inputs[i] = new NDArray(static_cast<void *>(data[i]), static_cast<Nd4jLong*>(inputShapeInfo[i]));

    sd::SpecialMethods<T>::concatCpuGeneric(inputs, output, dimension);

    for(int i = 0; i < numArrays; ++i)
        delete inputs[i];
}


template <typename T>
void SpecialMethods<T>::splitCpuGeneric(const NDArray& input, const std::vector<NDArray*>& outArrs, const int axis) {

    int numSplits = outArrs.size();

    const auto sizeofT = input.sizeOfT();

    auto xBuff = input.bufferAsT<T>();

    bool luckCase1 = ((axis == 0 && input.ordering() == 'c') || (axis == input.rankOf() - 1 && input.ordering() == 'f')) && input.ews() == 1;

    if (luckCase1) {
        for (uint i = 0; i < numSplits; ++i) {
            luckCase1 &= outArrs[i]->ordering() == input.ordering() && outArrs[i]->ews() == 1;
            if (!luckCase1)
                break;
        }
    }

    if (luckCase1) {

        T* x = const_cast<T*>(xBuff);
        for (uint i = 0; i < numSplits; ++i) {
            const auto memAmountToCopy = outArrs[i]->lengthOf();
            memcpy(outArrs[i]->bufferAsT<T>(), x, memAmountToCopy * sizeofT);
            x += memAmountToCopy;
        }
        return;
    }

    // const bool isXcontin = input.strideAt(axis) == 1;
    // bool areOutsContin = true;
    // bool allSameOrder = true;
    // std::vector<Nd4jLong> strideOfContigStride(numSplits);

    // if (isXcontin) {

    //     for (uint i = 0; i < numSplits; ++i) {

    //         areOutsContin &= outArrs[i]->strideAt(axis) == 1;
    //         allSameOrder &= outArrs[i]->ordering() == input.ordering();
    //         if (!areOutsContin || !allSameOrder)
    //             break;

    //         strideOfContigStride[i] = shape::strideOverContigAxis(axis, outArrs[i]->shapeInfo());
    //     }
    // }

    // const bool luckCase2 = isXcontin && areOutsContin && allSameOrder;

    // if (luckCase2) {

    //     const auto xStep = shape::strideOverContigAxis(axis, input.shapeInfo());

    //     for (uint i = 0; i < input.lengthOf() / input.sizeAt(axis); ++i) {

    //         T* x = xBuff + xStep * i;

    //         for (uint j = 0; j < numSplits; ++j) {
    //             const auto zDim = outArrs[j]->sizeAt(axis);
    //             T* z = outArrs[j]->bufferAsT<T>() + strideOfContigStride[j] * i;
    //             memcpy(z, x, zDim * sizeofT);
    //             x += zDim;
    //         }
    //     }

    //     return;
    // }

    uint zDim = outArrs[0]->sizeAt(axis);
    // general case

    auto func = PRAGMA_THREADS_FOR{

        int coords[MAX_RANK], temp;

        for (auto i = start; i < stop; i += increment) {

            shape::index2coordsCPU(start, i, input.shapeInfo(), coords);
            const auto xOffset = shape::getOffset(input.shapeInfo(), coords);

            uint outArrIdx = 0;
            temp = coords[axis];

            while (coords[axis] >= zDim) {
                coords[axis] -= zDim;
                ++outArrIdx;
            }

            T* z = outArrs[outArrIdx]->bufferAsT<T>();
            const auto zOffset = shape::getOffset(outArrs[outArrIdx]->shapeInfo(), coords);
            z[zOffset] = xBuff[xOffset];

            coords[axis] = temp;
        }
    };

    samediff::Threads::parallel_for(func, 0, input.lengthOf());
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
    void SpecialMethods<T>::accumulateGeneric(void **vx, void *vz, Nd4jLong const* zShapeInfo, int n, const Nd4jLong length) {
        auto z = reinterpret_cast<T *>(vz);
        auto x = reinterpret_cast<T **>(vx);

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
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
    void SpecialMethods<T>::averageGeneric(void **vx, void *vz, Nd4jLong const* zShapeInfo, int n, const Nd4jLong length, bool propagate) {
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
                for (auto i = start; i < stop; i++) {
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
                for (auto i = start; i < stop; i++) {
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
    Nd4jLong SpecialMethods<T>::getPosition(Nd4jLong const* xShapeInfo, Nd4jLong index) {
        auto xEWS = shape::elementWiseStride(xShapeInfo);

        if (xEWS == 1)
            return index;
        else if (xEWS > 1)
            return index * xEWS;
        else
            return shape::getIndexOffset(index, xShapeInfo);
    }

    template<typename T>
    void SpecialMethods<T>::quickSort_parallel_internal(T* array, Nd4jLong const* xShapeInfo, int left, int right, int cutoff, bool descending) {

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
    void SpecialMethods<T>::quickSort_parallel(void *varray, Nd4jLong const* xShapeInfo, Nd4jLong lenArray, int numThreads, bool descending){
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
    void SpecialMethods<T>::sortGeneric(void *vx, Nd4jLong const* xShapeInfo, bool descending) {
        auto x = reinterpret_cast<T *>(vx);

        quickSort_parallel(x, xShapeInfo, shape::length(xShapeInfo), omp_get_max_threads(), descending);
    }

    template<typename T>
    void SpecialMethods<T>::sortTadGeneric(void *vx, Nd4jLong const* xShapeInfo, int *dimension, int dimensionLength, Nd4jLong const* tadShapeInfo, Nd4jLong const* tadOffsets, bool descending) {
        auto x = reinterpret_cast<T *>(vx);

        //quickSort_parallel(x, xShapeInfo, shape::length(xShapeInfo), omp_get_max_threads(), descending);
        Nd4jLong xLength = shape::length(xShapeInfo);
        Nd4jLong xTadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
        int numTads = xLength / xTadLength;

        auto func = PRAGMA_THREADS_FOR {
            for (auto r = start; r < stop; r++) {
                T *dx = x + tadOffsets[r];

                quickSort_parallel(dx, tadShapeInfo, xTadLength, 1, descending);
            }
        };
        samediff::Threads::parallel_tad(func, 0, numTads);
    }


    template<typename T>
    void SpecialMethods<T>::decodeBitmapGeneric(const void *dx, Nd4jLong N, void *vz, Nd4jLong const* zShapeInfo) {
        auto dz = reinterpret_cast<T *>(vz);
        auto x = reinterpret_cast<const int *>(dx);
        Nd4jLong lim = N / 16 + 5;

        FloatBits2 fb;
        fb.i_ = x[2];
        float threshold = fb.f_;

        auto pPos = -1;

        auto func = PRAGMA_THREADS_FOR {
            for (auto e = start; e < stop; e++) {
                const auto v = x[e];
                for (int bitId = 0; bitId < 16; bitId++) {
                    bool hasBit = (v & 1 << (bitId)) != 0;
                    bool hasSign = (v & 1 << (bitId + 16)) != 0;
                    auto cPos = (e - 4) * 16 + bitId;

                    if (hasBit) {
                        if (hasSign)
                            dz[cPos] -= static_cast<T>(threshold);
                        else
                            dz[cPos] += static_cast<T>(threshold);
                    } else if (hasSign) {
                        dz[cPos] -= static_cast<T>(threshold / 2);
                    }

                    pPos = cPos;
                }
            }
        };

        samediff::Threads::parallel_for(func, 4, lim);
    }

    template<typename T>
    Nd4jLong SpecialMethods<T>::encodeBitmapGeneric(void *vx, Nd4jLong const* xShapeInfo, Nd4jLong N, int *dz, float threshold) {
        auto dx = reinterpret_cast<T *>(vx);
        const T two(2.0f);
        const T zero(0.0f);
        const T t(threshold);
        const T thalf = t / two;

        //auto func = PRAGMA_REDUCE_LONG {
            Nd4jLong retVal = 0L;

            PRAGMA_OMP_PARALLEL_FOR_REDUCTION(+:retVal)
            for (auto x = 0; x < N; x += 16) {
                int byte = 0;
                int byteId = x / 16 + 4;

                for (int f = 0; f < 16; f++) {
                    auto e = x + f;

                    if (e >= N)
                        continue;

                    T val = dx[e];
                    T abs = sd::math::nd4j_abs<T>(val);

                    int bitId = e % 16;

                    if (abs >= t) {
                        byte |= 1 << (bitId);
                        retVal++;

                        if (val < zero) {
                            byte |= 1 << (bitId + 16);
                            dx[e] += t;
                        } else {
                            dx[e] -= t;
                        }
                    } else if (abs >= thalf && val < zero) {
                        byte |= 1 << (bitId + 16);
                        dx[e] += thalf;

                        retVal++;
                    }
                }

                dz[byteId] = byte;
            }

            return retVal;
        //};

        //return samediff::Threads::parallel_long(func, LAMBDA_SUML, 0, N, 16);
    }
}

