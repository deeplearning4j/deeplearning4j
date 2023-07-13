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

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 12.06.2019
//
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/prefix.h>
#include <ops/ops.h>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void prefixPerBlockCuda(scalar::Ops op, const void* vx, const sd::LongType* xTadShapeInfo,
                                         const sd::LongType* xTadOffsets, void* vz, const sd::LongType* zTadShapeInfo,
                                         const sd::LongType* zTadOffsets, const sd::LongType numTads,
                                         const sd::LongType tadLen, const bool exclusive, const bool reverse) {
  __shared__ T *shared, lastElemInChunk;
  __shared__ sd::LongType numTadChunks, blockDim2;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    shared = reinterpret_cast<T*>(shmem);
    blockDim2 = 2 * blockDim.x;
    numTadChunks = (tadLen + blockDim2 - 1) / blockDim2;  // ceil
  }
  __syncthreads();

  const auto xTad = reinterpret_cast<const T*>(vx) + xTadOffsets[blockIdx.x];
  auto zTad = reinterpret_cast<T*>(vz) + zTadOffsets[blockIdx.x];

  sd::LongType sharedInd(2 * threadIdx.x), leftArrInd, rightArrInd, step;
  T xLeft, xRight;

  for (sd::LongType i = 0; i < numTadChunks; ++i) {
    leftArrInd = sharedInd + i * blockDim2;
    rightArrInd = leftArrInd + 1;

    if (reverse) {
      if (rightArrInd < tadLen) {
        rightArrInd = tadLen - 1 - rightArrInd;
        leftArrInd = tadLen - 1 - leftArrInd;
      } else if (leftArrInd < tadLen)
        leftArrInd = tadLen - 1 - leftArrInd;
    }

    if (leftArrInd < tadLen) shared[sharedInd] = xLeft = xTad[shape::getIndexOffset(leftArrInd, xTadShapeInfo)];
    if (rightArrInd < tadLen) shared[sharedInd + 1] = xRight = xTad[shape::getIndexOffset(rightArrInd, xTadShapeInfo)];

    step = 1;

    for (sd::LongType d = blockDim.x; d > 0; d /= 2) {
      __syncthreads();
      if (threadIdx.x < d) {
        sd::LongType left = step * (sharedInd + 1) - 1;
        sd::LongType right = step * (sharedInd + 2) - 1;
        shared[right] = (op == scalar::Add) ? (shared[right] + shared[left]) : (shared[right] * shared[left]);
      }
      step *= 2;
    }

    if (threadIdx.x == 0) shared[blockDim2 - 1] = (op == scalar::Add) ? 0 : 1;
    __syncthreads();

    for (sd::LongType d = 1; d < blockDim2; d *= 2) {
      step /= 2;

      __syncthreads();
      if (threadIdx.x < d) {
        sd::LongType left = step * (sharedInd + 1) - 1;
        sd::LongType right = step * (sharedInd + 2) - 1;
        T temp = shared[left];
        shared[left] = shared[right];
        shared[right] = (op == scalar::Add) ? (shared[right] + temp) : (shared[right] * temp);
      }
    }

    __syncthreads();

    if (leftArrInd < tadLen) {
      T result = shared[sharedInd];
      if (!exclusive) result = (op == scalar::Add) ? result + xLeft : result * xLeft;
      if (i > 0) result = (op == scalar::Add) ? result + lastElemInChunk : result * lastElemInChunk;
      zTad[shape::getIndexOffset(leftArrInd, zTadShapeInfo)] = result;
    }

    if (rightArrInd < tadLen) {
      T result = shared[sharedInd + 1];
      if (!exclusive) result = (op == scalar::Add) ? result + xRight : result * xRight;
      if (i > 0) result = (op == scalar::Add) ? result + lastElemInChunk : result * lastElemInChunk;
      if (i < numTadChunks - 1 && threadIdx.x == blockDim.x - 1)  // last element in chunk
        lastElemInChunk = !exclusive ? result : (op == scalar::Add) ? result + xRight : result * xRight;
      zTad[shape::getIndexOffset(rightArrInd, zTadShapeInfo)] = result;
    }
  }
}

///////////////////////////////////////////////////////////////////
template <typename X>
static void prefixPerBlockCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                       const cudaStream_t* stream, scalar::Ops op, const void* vx,
                                       const sd::LongType* xTadShapeInfo, const sd::LongType* xTadOffsets, void* vz,
                                       const sd::LongType* zTadShapeInfo, const sd::LongType* zTadOffsets,
                                       const sd::LongType numTads, const sd::LongType tadLen, const bool exclusive,
                                       const bool reverse) {
  prefixPerBlockCuda<X><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(
      op, vx, xTadShapeInfo, xTadOffsets, vz, zTadShapeInfo, zTadOffsets, numTads, tadLen, exclusive, reverse);
}

///////////////////////////////////////////////////////////////////
void prefix(sd::LaunchContext* context, scalar::Ops op, const NDArray* x, NDArray* z, const std::vector<LongType>& dims,
            bool exclusive, bool reverse) {
  auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), &dims);
  auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(), &dims);

  const sd::LongType numTads = packX->numberOfTads();
  const sd::LongType tadLen = x->lengthOf() / numTads;


  dim3 launchDims = prefixDims(numTads,x->sizeOfT());
  PointersManager manager(context, "prefix");

  NDArray::prepareSpecialUse({z}, {x});
  BUILD_SINGLE_SELECTOR(x->dataType(), prefixPerBlockCudaLauncher,
                        (launchDims.y, launchDims.x, launchDims.z, context->getCudaStream(), op, x->specialBuffer(),
                            packX->platformShapeInfo(), packX->platformOffsets(), z->specialBuffer(),
                            packZ->platformShapeInfo(), packZ->platformOffsets(), numTads, tadLen, exclusive, reverse),
                        SD_NUMERIC_TYPES);
  NDArray::registerSpecialUse({z}, {x});

  manager.synchronize();
}

///////////////////////////////////////////////////////////////////
void prefix(sd::LaunchContext* context, scalar::Ops op, const NDArray* x, NDArray* z, bool exclusive, bool reverse) {
  prefix(context, op, x, z, {}, exclusive, reverse);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
