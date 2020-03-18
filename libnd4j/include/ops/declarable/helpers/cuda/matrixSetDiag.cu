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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <array/ResultSet.h>
#include <ops/declarable/helpers/matrixSetDiag.h>
#include <helpers/PointersManager.h>

namespace sd    {
namespace ops     {
namespace helpers {

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void matrixSetDiagCuda(const void* vx, const Nd4jLong* xShapeInfo, const void* vy, const Nd4jLong* yShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const bool zeroPad) {

    // x - input,    shape [A,B,C]
    // y - diagonal, shape [A,B]
    // z - output,   shape [A,B,C]
    // input and output are the same array (x == z) when zeroPad = true

    const auto x = reinterpret_cast<const T*>(vx);
    const auto y = reinterpret_cast<const T*>(vy);
          auto z = reinterpret_cast<T*>(vz);

    __shared__ int xRank, *sharedMem;       // xRank = zRank, xRank = yRank + 1
    __shared__ Nd4jLong xLen;   // xLen = zLen
    __shared__ bool areSameOffsets;

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<int*>(shmem);

        areSameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);    // shapes are definitely the same, but strides might not

        xRank = shape::rank(xShapeInfo);
        xLen  = shape::length(xShapeInfo);
    }

    __syncthreads();

    auto coords = sharedMem + threadIdx.x * xRank;               // we provide (xRank * sizeof(int) * threadIdx.x) amount of shared memory per each thread
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < xLen; i += gridDim.x * blockDim.x) {

        shape::index2coords(i, xShapeInfo, coords);

        const auto xOffset = shape::getOffset(xShapeInfo, coords);
        const auto zOffset = areSameOffsets ? xOffset : shape::getOffset(zShapeInfo, coords);

        // condition to be on diagonal of innermost matrix
        if(coords[xRank - 2] == coords[xRank - 1])
            z[zOffset] = y[shape::getOffset(yShapeInfo, coords)];
        else
            z[zOffset] = zeroPad ? static_cast<T>(0) : x[xOffset];
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
static void matrixSetDiagCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,  const void* vx, const Nd4jLong* xShapeInfo, const void* vy, const Nd4jLong* yShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const bool zeroPad) {

    matrixSetDiagCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, zeroPad);
}

///////////////////////////////////////////////////////////////////
void matrixSetDiag(sd::LaunchContext* context, const NDArray& input, const NDArray& diagonal, NDArray& output, const bool zeroPad) {

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (input.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = threadsPerBlock * sizeof(int) * input.rankOf() + 128;

    PointersManager manager(context, "matrixSetDiag");

    NDArray::prepareSpecialUse({&output}, {&input, &diagonal});
    BUILD_SINGLE_SELECTOR(input.dataType(), matrixSetDiagCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), diagonal.getSpecialBuffer(), diagonal.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), zeroPad), LIBND4J_TYPES);
    NDArray::registerSpecialUse({&output}, {&input, &diagonal});

    manager.synchronize();
}

}
}
}