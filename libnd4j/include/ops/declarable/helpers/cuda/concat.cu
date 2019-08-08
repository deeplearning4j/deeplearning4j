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
            template<typename T>
            __global__ static void concatCuda(const int numOfArrs, void* pVx,  void* pxShapeInfo, void* pVz, void* pzShapeInfo) {

                __shared__ int arrIdx, blocksPerArr;

                if (threadIdx.x == 0) {

                    blocksPerArr = (gridDim.x + numOfArrs - 1) / numOfArrs;     // ceil
                    arrIdx = blockIdx.x / blocksPerArr;
                }

                __syncthreads();

                for(int j = arrIdx; j < numOfArrs; j += gridDim.x) {

                    const auto* x = reinterpret_cast<T*>(reinterpret_cast<void**>(pVx)[j]);
                    auto* z = reinterpret_cast<T*>(reinterpret_cast<void**>(pVz)[j]);
                    const auto* xShapeInfo = reinterpret_cast<Nd4jLong**>(pxShapeInfo)[j];
                    const auto* zShapeInfo = reinterpret_cast<Nd4jLong**>(pzShapeInfo)[j];

                    const auto arrLen = shape::length(xShapeInfo);

                    const auto arrLenPerBlock = (arrLen + blocksPerArr - 1) / blocksPerArr;  // ceil

                    const auto start = (blockIdx.x % blocksPerArr) * arrLenPerBlock;
                    const auto end   = (start + arrLenPerBlock) > arrLen ? arrLen : (start + arrLenPerBlock);

                    for (Nd4jLong i = start + threadIdx.x; i < end; i += blockDim.x)
                        z[shape::getIndexOffset(i, zShapeInfo, arrLen)] = x[shape::getIndexOffset(i, xShapeInfo, arrLen)];
                }
            }

///////////////////////////////////////////////////////////////////
            template<typename T>
            __host__ static void concatCudaLauncher(const int numOfArrs, const cudaStream_t *stream,  void* pVx, void* pxShapeInfo, void* pVz, void* pzShapeInfo) {

                concatCuda<T><<<512, 512, 512, *stream>>>(numOfArrs, pVx, pxShapeInfo, pVz, pzShapeInfo);
            }
            BUILD_SINGLE_TEMPLATE(template void concatCudaLauncher,  (const int numOfArrs, const cudaStream_t *stream, void* pVx, void* pxShapeInfo, void* pVz, void* pzShapeInfo), LIBND4J_TYPES);
        }
    }
}