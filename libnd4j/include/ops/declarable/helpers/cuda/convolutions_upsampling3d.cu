/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/convolutions.h>
#include <helpers/PointersManager.h>

namespace sd {
namespace ops  {

//////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ static void upsampling3dCuda(const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int factorD, const int factorH, const int factorW, const bool isNCDHW) {

    // x has shape [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC)
    // z has shape [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC] (NDHWC)

    const T* x = reinterpret_cast<const T*>(vx);
          T* z = reinterpret_cast<T*>(vz);

    __shared__ int rank, dimID;
    __shared__ Nd4jLong zLen, *sharedMem;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        dimID = isNCDHW ? 2 : 1;
        zLen  = shape::length(zShapeInfo);
        rank  = 5;
    }
    __syncthreads();

    const auto zInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(zInd >= zLen)
        return;

    auto coords = sharedMem + threadIdx.x * rank;

    shape::index2coords(zInd, zShapeInfo, coords);

    const auto zOffset = shape::getOffset(zShapeInfo, coords);

    coords[dimID]     /= factorD;
    coords[dimID + 1] /= factorH;
    coords[dimID + 2] /= factorW;

    const auto xOffset = shape::getOffset(xShapeInfo, coords);

    z[zOffset] = x[xOffset];
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void upsampling3dCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                     const void* vx, const Nd4jLong* xShapeInfo,
                                           void* vz, const Nd4jLong* zShapeInfo,
                                     const int factorD, const int factorH, const int factorW, const bool isNCDHW) {

    upsampling3dCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, factorD, factorH, factorW, isNCDHW);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::upsampling3d(sd::graph::Context& block, const NDArray& input, NDArray& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW) {

    PointersManager manager(block.launchContext(), "upsampling3d");

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = output.rankOf() * sizeof(Nd4jLong) * threadsPerBlock + 128;

    NDArray::prepareSpecialUse({&output}, {&input});
    BUILD_SINGLE_SELECTOR(input.dataType(), upsampling3dCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), input.specialBuffer(), input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), factorD, factorH, factorW, isNCDHW), FLOAT_TYPES);
    NDArray::registerSpecialUse({&output}, {&input});

    manager.synchronize();
}

}
}
