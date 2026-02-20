/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <ops/declarable/helpers/kv_scatter.h>
#include <array/NDArray.h>
#include <helpers/DebugHelper.h>
#include <cuda_runtime.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * CUDA kernel: copies present[b, h, lastPos, :] -> output[b, h, cachePos, :]
 *
 * Grid: one block per (batch, head) slice
 * Block: 256 threads, grid-stride over dim
 */
template <typename T>
__global__ void kvScatterKernel(const T* __restrict__ present,
                                 T* __restrict__ output,
                                 const LongType batch,
                                 const LongType heads,
                                 const LongType srcSeqLen,
                                 const LongType dstSeqLen,
                                 const LongType dim,
                                 const LongType lastPos,
                                 const LongType cachePos) {
    // Each block handles one (batch, head) slice
    auto slice = blockIdx.x;
    auto b = slice / heads;
    auto h = slice % heads;

    auto srcOffset = b * heads * srcSeqLen * dim + h * srcSeqLen * dim + lastPos * dim;
    auto dstOffset = b * heads * dstSeqLen * dim + h * dstSeqLen * dim + cachePos * dim;

    // Grid-stride loop over dim
    for (LongType d = threadIdx.x; d < dim; d += blockDim.x) {
        output[dstOffset + d] = present[srcOffset + d];
    }
}

template <typename T>
static void kvScatterCudaLauncher(const cudaStream_t* stream,
                                   const void* vPresent, void* vOutput,
                                   LongType batch, LongType heads,
                                   LongType srcSeqLen, LongType dstSeqLen,
                                   LongType dim, LongType lastPos,
                                   LongType cachePos) {
    auto present = reinterpret_cast<const T*>(vPresent);
    auto output = reinterpret_cast<T*>(vOutput);

    auto numSlices = batch * heads;
    int threads = 256;

    kvScatterKernel<T><<<numSlices, threads, 0, *stream>>>(
        present, output, batch, heads, srcSeqLen, dstSeqLen, dim, lastPos, cachePos);
    DebugHelper::checkGlobalErrorCode("kvScatter kernel failed");
}

BUILD_SINGLE_TEMPLATE(void kvScatterCudaLauncher, (const cudaStream_t* stream, const void* vPresent, void* vOutput, LongType batch, LongType heads, LongType srcSeqLen, LongType dstSeqLen, LongType dim, LongType lastPos, LongType cachePos), SD_FLOAT_TYPES);

void kvScatter(NDArray* present, NDArray* output,
               LongType cachePos, LaunchContext* context) {
    auto stream = context->getCudaStream();

    auto batch = present->sizeAt(0);
    auto heads = present->sizeAt(1);
    auto srcSeqLen = present->sizeAt(2);
    auto dstSeqLen = output->sizeAt(2);
    auto dim = present->sizeAt(3);
    auto lastPos = srcSeqLen - 1;

    NDArray::prepareSpecialUse({output}, {present});

    BUILD_SINGLE_SELECTOR(present->dataType(), kvScatterCudaLauncher,
                          (stream, present->specialBuffer(), output->specialBuffer(),
                           batch, heads, srcSeqLen, dstSeqLen, dim, lastPos, cachePos),
                          SD_FLOAT_TYPES);

    NDArray::registerSpecialUse({output}, {present});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
