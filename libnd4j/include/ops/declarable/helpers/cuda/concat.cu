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
                __shared__ T *x, *z;
                __shared__ Nd4jLong *zShapeInfo, *xShapeInfo, arrLen, arrLenPerBlock, start, end;

                if (threadIdx.x == 0) {

                    blocksPerArr = (gridDim.x + numOfArrs - 1) / numOfArrs;     // ceil
                    arrIdx = blockIdx.x / blocksPerArr;

                    x = reinterpret_cast<T*>(reinterpret_cast<void**>(pVx)[arrIdx]);
                    z = reinterpret_cast<T*>(reinterpret_cast<void**>(pVz)[arrIdx]);
                    xShapeInfo = reinterpret_cast<Nd4jLong**>(pxShapeInfo)[arrIdx];
                    zShapeInfo = reinterpret_cast<Nd4jLong**>(pzShapeInfo)[arrIdx];
                    arrLen = shape::length(xShapeInfo);

                    arrLenPerBlock = (arrLen + blocksPerArr - 1) / blocksPerArr;  // ceil

                    start = (blockIdx.x % blocksPerArr) * arrLenPerBlock;
                    end   = (start + arrLenPerBlock) > arrLen ? arrLen : (start + arrLenPerBlock);
                }

                __syncthreads();

                for (Nd4jLong i = start + threadIdx.x; i < end; i += blockDim.x)
                    z[shape::getIndexOffset(i, zShapeInfo, arrLen)] = x[shape::getIndexOffset(i, xShapeInfo, arrLen)];
            }

///////////////////////////////////////////////////////////////////
            template<typename T>
            __host__ static void concatCudaLauncher(const int numOfArrs, const cudaStream_t *stream,  void* pVx, void* pxShapeInfo, void* pVz, void* pzShapeInfo) {

                concatCuda<T><<<512, 256, 1024, *stream>>>(numOfArrs, pVx, pxShapeInfo, pVz, pzShapeInfo);
            }
            BUILD_SINGLE_TEMPLATE(template void concatCudaLauncher,  (const int numOfArrs, const cudaStream_t *stream, void* pVx, void* pxShapeInfo, void* pVz, void* pzShapeInfo), LIBND4J_TYPES);

            //////////////////////////////////////////////////////////////////////////
            void concat(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output, const int axis) {

                const int numOfArrs = inArrs.size();
                for(int i = 0; i < numOfArrs; ++i)
                    if(!inArrs[i]->isActualOnDeviceSide()) inArrs[i]->syncToDevice();

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

                std::vector<NDArray*> outSubArrs(numOfArrs);
                for(int i = 0; i < numOfArrs; ++i)
                    outSubArrs[i] = new NDArray(output(indices[i], true));

                // prepare arrays of pointers on buffers and shapes
                std::vector<void*>     hOutBuffers(numOfArrs), hInBuffers(numOfArrs);
                std::vector<Nd4jLong*> hOutShapeInfo(numOfArrs), hInShapeInfo(numOfArrs);
                for(int i = 0; i < numOfArrs; ++i) {
                    hOutBuffers[i]   = outSubArrs[i]->getSpecialBuffer();
                    hInBuffers[i]    =     inArrs[i]->getSpecialBuffer();
                    hOutShapeInfo[i] = outSubArrs[i]->getSpecialShapeInfo();
                    hInShapeInfo[i]  =     inArrs[i]->getSpecialShapeInfo();
                }

                // allocate and copy all buffers and shapes arrays to global memory
                PointersManager manager(context, "helpers::concat");
                void* dOutBuffers	= manager.replicatePointer(hOutBuffers.data(),   hOutBuffers.size() * sizeof(void*));
                void* dInBuffers	= manager.replicatePointer(hInBuffers.data(),    hInBuffers.size() * sizeof(void*));
                void* dInShapeInfo  = manager.replicatePointer(hInShapeInfo.data(),  hInShapeInfo.size() * sizeof(Nd4jLong*));
                void* dOutShapeInfo = manager.replicatePointer(hOutShapeInfo.data(), hOutShapeInfo.size() * sizeof(Nd4jLong*));

                BUILD_SINGLE_SELECTOR(inArrs[0]->dataType(), concatCudaLauncher, (numOfArrs, context->getCudaStream(), dInBuffers, dInShapeInfo, dOutBuffers, dOutShapeInfo), LIBND4J_TYPES);

                manager.synchronize();

                for(int i = 0; i < numOfArrs; ++i)
                    delete outSubArrs[i];

                for(int i = 0; i < numOfArrs; ++i)
                    inArrs[i]->tickReadHost();

                output.tickWriteDevice();
            }
        }
    }
}