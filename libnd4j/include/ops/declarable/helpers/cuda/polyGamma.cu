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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 26.04.2019
//

#include<ops/declarable/helpers/gammaMathFunc.h>
#include<ops/declarable/helpers/zeta.h>
#include <array/NDArrayFactory.h>

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void polyGammaCuda(const void *vn, const Nd4jLong *nShapeInfo,
                                	 const void *vx, const Nd4jLong *xShapeInfo,
                                     	   void *vz, const Nd4jLong *zShapeInfo) {

    const auto n = reinterpret_cast<const T*>(vn);
    const auto x = reinterpret_cast<const T*>(vx);
          auto z = reinterpret_cast<T*>(vz);

    __shared__ Nd4jLong len;
    __shared__ bool sameOffsetNX, sameOffsetNZ;

    if (threadIdx.x == 0) {
        len = shape::length(nShapeInfo);
        sameOffsetNX = shape::haveSameShapeAndStrides(xShapeInfo, nShapeInfo);
        sameOffsetNZ = shape::haveSameShapeAndStrides(zShapeInfo, nShapeInfo);
    }
    __syncthreads();

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto totalThreads = gridDim.x * blockDim.x;

    for (int i = tid; i < len; i += totalThreads) {

        const auto nOffset = shape::getIndexOffset(i, nShapeInfo);
        const auto xOffset = sameOffsetNX ? nOffset : shape::getIndexOffset(i, xShapeInfo);
        const auto zOffset = sameOffsetNZ ? nOffset : shape::getIndexOffset(i, zShapeInfo);

        const T order = n[nOffset];

        int sign = (static_cast<int>(order) + 1) % 2  ?  -1 : 1;

        if(order != static_cast<int>(order)) {
            z[zOffset] = DataTypeUtils::nanOrZero<T>();
        }
        else if(order == 0) {
            z[zOffset] = diGammaScalar<T>(x[xOffset]);
        }
        else {
            T factorial = 1;
            for(int i = 2; i <= order; ++i)
                factorial *= i;

            z[zOffset] = sign * factorial * zetaScalar<T>(order + 1, x[xOffset]);
        }
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
static void polyGammaCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void *vn, const Nd4jLong *nShapeInfo, const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo) {

    polyGammaCuda<T><<<blocksPerGrid, threadsPerBlock, 1024, *stream>>>(vn, nShapeInfo, vx, xShapeInfo, vz, zShapeInfo);
}

///////////////////////////////////////////////////////////////////
void polyGamma(sd::LaunchContext * context, const NDArray& n, const NDArray& x, NDArray& z) {

    NDArray::prepareSpecialUse({&z}, {&n, &x});

    int threadsPerBlock = MAX_NUM_THREADS / 2;
    int blocksPerGrid = (z.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    BUILD_SINGLE_SELECTOR(n.dataType(), polyGammaCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), n.specialBuffer(), n.specialShapeInfo(), x.specialBuffer(), x.specialShapeInfo(), z.specialBuffer(), z.specialShapeInfo()), FLOAT_TYPES);

    NDArray::registerSpecialUse({&z}, {&n, &x});
}

BUILD_SINGLE_TEMPLATE(template void polyGammaCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void *vn, const Nd4jLong *nShapeInfo, const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo), FLOAT_TYPES);

}
}
}

