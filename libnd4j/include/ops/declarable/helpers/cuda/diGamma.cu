/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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

#include<ops/declarable/helpers/gammaMathFunc.h>
#include <array/NDArrayFactory.h>

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void diGammaCuda(const void *vx, const Nd4jLong *xShapeInfo,
                                     	 void *vz, const Nd4jLong *zShapeInfo) {

    const auto x = reinterpret_cast<const T*>(vx);
          auto z = reinterpret_cast<T*>(vz);

    __shared__ Nd4jLong len;
    __shared__ bool sameOffset;

    if (threadIdx.x == 0) {
        len = shape::length(xShapeInfo);
        sameOffset = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
    }
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x) {

        const auto xOffset = shape::getIndexOffset(i, xShapeInfo);
        const auto zOffset = sameOffset ? xOffset : shape::getIndexOffset(i, zShapeInfo);

        z[zOffset] = diGammaScalar<T>(x[xOffset]);
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
static void diGammaCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo) {

    diGammaCuda<T><<<blocksPerGrid, threadsPerBlock, 1024, *stream>>>(vx, xShapeInfo, vz, zShapeInfo);
}

///////////////////////////////////////////////////////////////////
void diGamma(sd::LaunchContext* context, const NDArray& x, NDArray& z) {

    int threadsPerBlock = MAX_NUM_THREADS / 2;
    int blocksPerGrid = (z.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    NDArray::prepareSpecialUse({&z}, {&x});
    BUILD_SINGLE_SELECTOR(x.dataType(), diGammaCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), x.specialBuffer(), x.specialShapeInfo(), z.specialBuffer(), z.specialShapeInfo()), FLOAT_TYPES);
    NDArray::registerSpecialUse({&z}, {&x});
}

BUILD_SINGLE_TEMPLATE(template void diGammaCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo), FLOAT_TYPES);

}
}
}

