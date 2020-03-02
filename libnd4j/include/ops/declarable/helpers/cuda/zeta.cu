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

#include<ops/declarable/helpers/zeta.h>

namespace sd {
namespace ops {
namespace helpers {


///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void zetaCuda(const void *vx, const Nd4jLong *xShapeInfo,
                                const void *vq, const Nd4jLong *qShapeInfo,
                                      void *vz, const Nd4jLong *zShapeInfo) {

    const auto x = reinterpret_cast<const T*>(vx);
    const auto q = reinterpret_cast<const T*>(vq);
          auto z = reinterpret_cast<T*>(vz);

    __shared__ Nd4jLong len;

    if (threadIdx.x == 0)
        len = shape::length(xShapeInfo);
    __syncthreads();

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto totalThreads = gridDim.x * blockDim.x;

    for (int i = tid; i < len; i += totalThreads) {

        const auto xOffset = shape::getIndexOffset(i, xShapeInfo);
        const auto qOffset = shape::getIndexOffset(i, qShapeInfo);
        const auto zOffset = shape::getIndexOffset(i, zShapeInfo);

        z[zOffset] = zetaScalar<T>(x[xOffset], q[qOffset]);
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
static void zetaCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, const void *vq, const Nd4jLong *qShapeInfo, void *vz, const Nd4jLong *zShapeInfo) {

    zetaCuda<T><<<blocksPerGrid, threadsPerBlock, 1024, *stream>>>(vx, xShapeInfo, vq, qShapeInfo, vz, zShapeInfo);
}

void zeta(sd::LaunchContext * context, const NDArray& x, const NDArray& q, NDArray& z) {

    if(!x.isActualOnDeviceSide()) x.syncToDevice();
    if(!q.isActualOnDeviceSide()) q.syncToDevice();

    int threadsPerBlock = MAX_NUM_THREADS / 2;
    int blocksPerGrid = (z.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    BUILD_SINGLE_SELECTOR(x.dataType(), zetaCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), x.getSpecialBuffer(), x.getSpecialShapeInfo(), q.getSpecialBuffer(), q.getSpecialShapeInfo(), z.getSpecialBuffer(), z.getSpecialShapeInfo()), FLOAT_TYPES);

    x.tickReadHost();
    q.tickReadHost();
    z.tickWriteDevice();
}

BUILD_SINGLE_TEMPLATE(template void zetaCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, const void *vq, const Nd4jLong *qShapeInfo, void *vz, const Nd4jLong *zShapeInfo), FLOAT_TYPES);


}
}
}

