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
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/convolutions.h>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void upsampling2dCuda(const void* vx, const LongType* xShapeInfo, void* vz,
                                       const LongType* zShapeInfo, const LongType factorH, const LongType factorW,
                                       const bool isNCHW) {
  // x has shape [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC)
  // z has shape [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)

  const T* x = reinterpret_cast<const T*>(vx);
  T* z = reinterpret_cast<T*>(vz);

  __shared__ LongType rank, dimIH;
  __shared__ LongType zLen, *sharedMem;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    dimIH = isNCHW ? 2 : 1;
    zLen = shape::length(zShapeInfo);
    rank = 4;
  }
  __syncthreads();

  const auto zInd = threadIdx.x + blockIdx.x * blockDim.x;

  if (zInd >= zLen) return;

  auto coords = sharedMem + threadIdx.x * rank;

  INDEX2COORDS(zInd, rank, shape::shapeOf(zShapeInfo), coords);

  LongType zOffset;
  COORDS2INDEX(rank, shape::stride(zShapeInfo), coords, zOffset);

  coords[dimIH] /= factorH;
  coords[dimIH + 1] /= factorW;

  LongType xOffset;
  COORDS2INDEX(rank, shape::stride(xShapeInfo), coords, xOffset);

  z[zOffset] = x[xOffset];
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void upsampling2dCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                     const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                                     void* vz, const LongType* zShapeInfo, const LongType factorH, const LongType factorW,
                                     const bool isNCHW) {
  upsampling2dCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, factorH,
                                                                              factorW, isNCHW);
  DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream),"upsampling2dCudaLauncher failed");

}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::upsampling2d(graph::Context& block, NDArray& input, NDArray& output, const LongType factorH,
                                    const LongType factorW, const bool isNCHW) {
  PointersManager manager(block.launchContext(), "upsampling2d");

  dim3 getUpSampling = getUpsamplingDims(output.lengthOf(),output.rankOf());
  NDArray::prepareSpecialUse({&output}, {&input});
  BUILD_SINGLE_SELECTOR(
      input.dataType(), upsampling2dCudaLauncher,
      (getUpSampling.x, getUpSampling.y, getUpSampling.z, block.launchContext()->getCudaStream(), input.specialBuffer(),
       input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), factorH, factorW, isNCHW),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&output}, {&input});

  manager.synchronize();
}

}  // namespace ops
}  // namespace sd
