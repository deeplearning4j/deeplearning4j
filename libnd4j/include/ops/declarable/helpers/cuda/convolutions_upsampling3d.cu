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
SD_KERNEL static void upsampling3dCuda(const void* vx, const LongType* xShapeInfo, void* vz,
                                       const LongType* zShapeInfo, const int factorD, const int factorH,
                                       const int factorW, const bool isNCDHW) {
  // x has shape [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC)
  // z has shape [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC]
  // (NDHWC)

  const T* x = reinterpret_cast<const T*>(vx);
  T* z = reinterpret_cast<T*>(vz);

  __shared__ int rank, dimID;
  __shared__ LongType zLen, *sharedMem;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    dimID = isNCDHW ? 2 : 1;
    zLen = shape::length(zShapeInfo);
    rank = 5;
  }
  __syncthreads();

  const auto zInd = threadIdx.x + blockIdx.x * blockDim.x;

  if (zInd >= zLen) return;

  auto coords = sharedMem + threadIdx.x * rank;

  INDEX2COORDS(zInd, rank, shape::shapeOf(zShapeInfo), coords);

  LongType zOffset;
  COORDS2INDEX(rank, shape::stride(zShapeInfo), coords, zOffset);

  coords[dimID] /= factorD;
  coords[dimID + 1] /= factorH;
  coords[dimID + 2] /= factorW;

  LongType xOffset;
  COORDS2INDEX(rank, shape::stride(xShapeInfo), coords, xOffset);

  z[zOffset] = x[xOffset];
}
//////////////////////////////////////////////////////////////////////////
template <typename T>
static void upsampling3dCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                     const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                                     void* vz, const LongType* zShapeInfo, const int factorD, const int factorH,
                                     const int factorW, const bool isNCDHW) {
  upsampling3dCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, factorD,
                                                                              factorH, factorW, isNCDHW);

  DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream),"upsampling3dCudaLauncher failed");

}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::upsampling3d(graph::Context& block, NDArray& input, NDArray& output, const LongType factorD,
                                    const LongType factorH, const LongType factorW, const bool isNCDHW) {
  PointersManager manager(block.launchContext(), "upsampling3d");


  dim3 getUpSampling = getUpsamplingDims(output.lengthOf(),output.rankOf());

  NDArray::prepareSpecialUse({&output}, {&input});
  BUILD_SINGLE_SELECTOR(
      input.dataType(), upsampling3dCudaLauncher,
      (getUpSampling.x, getUpSampling.y, getUpSampling.z, block.launchContext()->getCudaStream(), input.specialBuffer(),
       input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), factorD, factorH, factorW, isNCDHW),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&output}, {&input});

  manager.synchronize();
}

}  // namespace ops
}  // namespace sd
