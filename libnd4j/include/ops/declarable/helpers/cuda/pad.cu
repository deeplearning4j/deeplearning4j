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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//


#include<ops/declarable/helpers/transforms.h>
#include <array/ResultSet.h>
#include <helpers/ShapeUtils.h>
#include <numeric>
#include <NDArrayFactory.h>
#include <helpers/TAD.h>
#include <exceptions/cuda_exception.h>
#include <PointersManager.h>
#include <ConstantTadHelper.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
///////////////////////////////////////////////////////////////////
// x - input, y - paddings, z - output
            template<typename X, typename Y>
            __global__ static void padCuda(const int mode,
                                           const void *vx, const Nd4jLong *xShapeInfo,
                                           const void *vy, const Nd4jLong *yShapeInfo,
                                           void *vz, const Nd4jLong *zShapeInfo,
                                           const void *vPadVal) {

                const X padVal = *reinterpret_cast<const X*>(vPadVal);

                const auto x = reinterpret_cast<const X*>(vx);
                const auto y = reinterpret_cast<const Y*>(vy);
                auto z = reinterpret_cast<X*>(vz);

                __shared__ int rank, rankMinusOne;
                __shared__ Nd4jLong zLen, yLen, totalThreads, *coords, *xShape, *zShape, *xStride, *zStride, shift1, shift2, yStride0;

                if (threadIdx.x == 0) {
                    extern __shared__ unsigned char shmem[];
                    coords    = reinterpret_cast<Nd4jLong*>(shmem);
                    zLen     = shape::length(zShapeInfo);
                    xShape   = shape::shapeOf(const_cast<Nd4jLong*>(xShapeInfo));
                    zShape   = shape::shapeOf(const_cast<Nd4jLong*>(zShapeInfo));
                    xStride  = shape::stride(const_cast<Nd4jLong*>(xShapeInfo));
                    zStride  = shape::stride(const_cast<Nd4jLong*>(zShapeInfo));
                    yStride0 = shape::stride(const_cast<Nd4jLong*>(yShapeInfo))[0];
                    rank     = shape::rank(xShapeInfo);
                    zLen     = shape::length(zShapeInfo);
                    yLen     = 2 * rank;
                    rankMinusOne = rank - 1;
                    totalThreads = gridDim.x * blockDim.x;
                    shift1 = mode == 1 ? 0 : 1;         // REFLECT : SYMMETRIC
                    shift2 = mode == 1 ? 2 : 1;         // REFLECT : SYMMETRIC
                }

                __syncthreads();

                auto xzCoord = coords + threadIdx.x * rank;       // we use xzCoord storage both for x and z arrays

                const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

                if(mode == 0) { // CONSTANT case

                    for (Nd4jLong i = tid; i < zLen; i += totalThreads) {

                        shape::index2coords(rank, zShape, i, zLen, xzCoord);
                        const auto zOffset = shape::getOffset(0, zShape, zStride, xzCoord, rank);

                        bool within = true;
                        for(int j = rankMinusOne; j >= 0; --j) {
                            if(xShape[j] == zShape[j]) continue;
                            const auto left = y[shape::getIndexOffset(yStride0 * j, yShapeInfo, yLen)];
                            if(xzCoord[j] < left || xzCoord[j] >= left + xShape[j]) {within = false; break;}
                            else                                                    {xzCoord[j] = xzCoord[j] - left;}
                        }

                        if(within)
                            z[zOffset] = x[shape::getOffset(0, xShape, xStride, xzCoord, rank)];
                        else
                            z[zOffset] = padVal;
                    }
                }
                else {  // REFLECT and SYMMETRIC cases

                    for (Nd4jLong i = tid; i < zLen; i += totalThreads) {

                        shape::index2coords(rank, zShape, i, zLen, xzCoord);
                        const auto zOffset = shape::getOffset(0, zShape, zStride, xzCoord, rank);

                        for(int j = rankMinusOne; j >= 0; --j) {

                            if(xShape[j] == zShape[j]) continue;
                            xzCoord[j] = xzCoord[j] - y[shape::getIndexOffset(yStride0 * j, yShapeInfo, yLen)];    // are ready to fill middle (within input dimension range)
                            if(xzCoord[j] < 0)               xzCoord[j] = -xzCoord[j] - shift1;                // means fill from left
                            else if(xzCoord[j] >= xShape[j]) xzCoord[j] = 2 * xShape[j] - xzCoord[j] - shift2; // means fill from right
                        }

                        const auto xOffset = shape::getOffset(0, xShape, xStride, xzCoord, rank);
                        z[zOffset] = x[xOffset];
                    }
                }
            }

///////////////////////////////////////////////////////////////////
            template<typename X, typename Y>
            static void padCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                        const int mode,
                                        const void *vx, const Nd4jLong *xShapeInfo,
                                        const void *vy, const Nd4jLong *yShapeInfo,
                                        void *vz, const Nd4jLong *zShapeInfo,
                                        const void* padVal) {

                padCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(mode, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, padVal);
            }

///////////////////////////////////////////////////////////////////
            void pad(nd4j::LaunchContext * context, const int mode, const NDArray& input, const NDArray& paddings, NDArray& output, const NDArray& padValue) {

                PointersManager manager(context, "pad");

                NDArray::prepareSpecialUse({&output}, {&input, &paddings, &padValue});

                const int threadsPerBlock = MAX_NUM_THREADS / 4;
                const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
                const int sharedMem = 8 * threadsPerBlock * output.rankOf() + 128;

                const auto xType = input.dataType();
                const auto yType = paddings.dataType();

                BUILD_DOUBLE_SELECTOR(xType, yType, padCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), mode, input.getSpecialBuffer(), input.getSpecialShapeInfo(), paddings.getSpecialBuffer(), paddings.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), padValue.getSpecialBuffer()), LIBND4J_TYPES, INDEXING_TYPES);

                NDArray::registerSpecialUse({&output}, {&input, &paddings, &padValue});
                manager.synchronize();
            }


            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            template <typename T>
            static __global__ void mirrorPadLinearKernel(void const* vx, Nd4jLong* xShape, void* vz, Nd4jLong* zShape, Nd4jLong leftSide, Nd4jLong leftSideCorrected, Nd4jLong xLen, Nd4jLong len, Nd4jLong zLen) {

                __shared__ T const* x;
                __shared__ T* z;
                if (threadIdx.x == 0) {
                    x = reinterpret_cast<T const*>(vx);
                    z = reinterpret_cast<T*>(vz);
                }
                __syncthreads();
                auto start = blockIdx.x * blockDim.x + threadIdx.x;
                auto step = blockDim.x * gridDim.x;

                for(int i = start; i < zLen; i+= step) {
                    auto zIndex = shape::getIndexOffset(i, zShape, zLen);
                    auto xIndex = shape::getIndexOffset(len - i, xShape, xLen);

                    if (i < leftSide)                                   // left side
                        xIndex = shape::getIndexOffset(leftSideCorrected - i, xShape, xLen);

                    else if(i >= leftSide && i < leftSide + xLen)       // middle
                        xIndex = shape::getIndexOffset(i - leftSide, xShape, xLen);

//            else                                                // right side
//                z[i] = x[len - i];
                    z[zIndex] = x[xIndex];
                }

            }

            template <typename F, typename I>
            static __global__ void mirrorPadKernel(void const* vx, Nd4jLong* xShape, void* vz, Nd4jLong* zShape, Nd4jLong outLen, void const* paddings, Nd4jLong* paddingShape, int reflBorder) {

                __shared__ F const* x;
                __shared__ I const* pads;
                __shared__ F* z;
                __shared__ Nd4jLong zRank, rank;
                __shared__ Nd4jLong* xShapeOf, *xStrideOf, *padsShapeOf, *padsStrideOf;
                __shared__ Nd4jLong* zShapeOf, *zStrideOf;
                __shared__ Nd4jLong* xIdx;
                if (threadIdx.x == 0) {
                    extern __shared__ unsigned char shmem[];
                    xIdx    = reinterpret_cast<Nd4jLong*>(shmem);
                    rank = shape::rank(xShape);

                    x = reinterpret_cast<F const*>(vx);//
                    pads = reinterpret_cast<I const*>(paddings);
                    z = reinterpret_cast<F*>(vz);
                    xShapeOf = shape::shapeOf(xShape);
                    xStrideOf = shape::stride(xShape);
                    zShapeOf = shape::shapeOf(zShape);
                    zRank = shape::rank(zShape);
                    zStrideOf = shape::stride(zShape);
                    padsShapeOf = shape::shapeOf(paddingShape);
                    padsStrideOf = shape::stride(paddingShape);
                }
                __syncthreads();
                auto start = threadIdx.x + blockIdx.x * blockDim.x;
                auto step = blockDim.x * gridDim.x;

                for(Nd4jLong i = start; i < outLen; i+= step) {
                    auto xzCoord = xIdx + threadIdx.x * rank;
                    //auto zxCoord = xIdx + (threadIdx.x + threadIdx.x % 2 + 1) * rank;

                    shape::index2coords(rank, zShapeOf, i, xzCoord);
                    auto outOffset = shape::getOffset(0, zShapeOf, zStrideOf, xzCoord, rank);
//                auto intStep = blockDim.y * gridDim.y;
                    for(int j = 0; j < rank; j++) {

                        const Nd4jLong inLen         = shape::sizeAt(xShape, j);
                        Nd4jLong coords[2] = {j, 0};
                        auto padOffset = shape::getOffset(0, padsShapeOf, padsStrideOf, coords, 2); // padding already has rank 2
                        const auto leftSide          = pads[padOffset];
                        const auto leftSideCorrected = leftSide - reflBorder;
                        const Nd4jLong len           = 2 * (inLen - 1) + leftSide + reflBorder;

                        if(xzCoord[j] < leftSide)                                        // left side
                            xzCoord[j] = leftSideCorrected - xzCoord[j];

                        else if(xzCoord[j] >= leftSide && xzCoord[j] < leftSide + inLen)  // middle
                            xzCoord[j] = xzCoord[j] - leftSide;

                        else if (len > xzCoord[j])                                                           // right side
                            xzCoord[j] = len - xzCoord[j];
                        else
                            xzCoord[j] = xzCoord[j] - len;
                    }

                    auto inOffset  = shape::getOffset(0, xShapeOf, xStrideOf,  xzCoord,  rank);
                    z[outOffset] = x[inOffset];
                }
            }

            template<typename F, typename I>
            static void mirrorPad_(nd4j::LaunchContext * context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode) {
                // mode:  0 - REFLECT, else - SYMMETRIC
                const int reflBorder = (bool)mode ? 1 : 0;
                const int rank        = input.rankOf();
                const Nd4jLong outLen = output.lengthOf();
                auto stream = context->getCudaStream();
                NDArray::prepareSpecialUse({&output}, {&input, &paddings});

                if(rank <= 1) {

                    const Nd4jLong inLen         = input.lengthOf();
                    const auto leftSide          = paddings.e<Nd4jLong>(0);
                    const auto leftSideCorrected = leftSide - reflBorder;
                    const Nd4jLong len           = 2*(inLen-1) + leftSide + reflBorder;

                    mirrorPadLinearKernel<F><<<256, 512, 256, *stream>>>(input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), leftSide, leftSideCorrected, inLen, len, outLen);
                    nd4j::DebugHelper::checkErrorCode(stream, "helpers::mirrorPadLinearKernel(...) failed");
                }
                else {
                    mirrorPadKernel<F, I><<<256, 256, 8192, *stream>>>(input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), outLen, paddings.getSpecialBuffer(), paddings.getSpecialShapeInfo(), reflBorder);
                    nd4j::DebugHelper::checkErrorCode(stream, "helpers::mirrorPadKernel(...) failed");
                }
                NDArray::registerSpecialUse({&output}, {&input, &paddings});
            }

            void mirrorPad(nd4j::LaunchContext * context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode) {
                BUILD_DOUBLE_SELECTOR(input.dataType(), paddings.dataType(), mirrorPad_, (context, input, paddings, output, mode), LIBND4J_TYPES, INDEXING_TYPES);
            }


        }
    }
}