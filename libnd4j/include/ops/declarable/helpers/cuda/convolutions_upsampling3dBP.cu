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

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void upsampling3dBPCuda(const void* vx, const sd::LongType* xShapeInfo, void* vz,
                                         const sd::LongType* zShapeInfo, const bool isNCDHW) {
  // x (gradO) has shape [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC)
  // z (gradI) has shape [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH,
  // factorW*iW, iC] (NDHWC)

  const T* x = reinterpret_cast<const T*>(vx);
  T* z = reinterpret_cast<T*>(vz);

  __shared__ int rank, dimID;
  __shared__ sd::LongType factorD, factorH, factorW;
  __shared__ sd::LongType zLen, *sharedMem;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<sd::LongType*>(shmem);

    dimID = isNCDHW ? 2 : 1;
    zLen = shape::length(zShapeInfo);
    rank = 5;

    factorD = xShapeInfo[dimID + 1] / zShapeInfo[dimID + 1];
    factorH = xShapeInfo[dimID + 2] / zShapeInfo[dimID + 2];
    factorW = xShapeInfo[dimID + 3] / zShapeInfo[dimID + 3];
  }
  __syncthreads();

  const auto zInd = threadIdx.x + blockIdx.x * blockDim.x;

  if (zInd >= zLen) return;

  auto coords = sharedMem + threadIdx.x * rank;

  shape::index2coords(zInd, zShapeInfo, coords);

  const auto zOffset = shape::getOffset(zShapeInfo, coords);

  z[zOffset] = 0;

  const sd::LongType zCoord2 = coords[dimID] * factorD;
  const sd::LongType zCoord3 = coords[dimID + 1] * factorH;
  const sd::LongType zCoord4 = coords[dimID + 2] * factorW;

  for (coords[dimID] = zCoord2; coords[dimID] < zCoord2 + factorD; ++coords[dimID])
    for (coords[dimID + 1] = zCoord3; coords[dimID + 1] < zCoord3 + factorH; ++coords[dimID + 1])
      for (coords[dimID + 2] = zCoord4; coords[dimID + 2] < zCoord4 + factorW; ++coords[dimID + 2])
        z[zOffset] += x[shape::getOffset(xShapeInfo, coords)];
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void upsampling3dBPCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                       const cudaStream_t* stream, const void* vx, const sd::LongType* xShapeInfo,
                                       void* vz, const sd::LongType* zShapeInfo, const bool isNCDHW) {
  upsampling3dBPCuda<T>
      <<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, isNCDHW);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::upsampling3dBP(sd::graph::Context& block, const NDArray& gradO, NDArray& gradI,
                                      const bool isNCDHW) {
  PointersManager manager(block.launchContext(), "upsampling3d_bp");

  const int threadsPerBlock = SD_MAX_NUM_THREADS / 2;
  const int blocksPerGrid = (gradI.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
  const int sharedMem = gradI.rankOf() * sizeof(sd::LongType) * threadsPerBlock + 128;

  NDArray::prepareSpecialUse({&gradI}, {&gradO});
  BUILD_SINGLE_SELECTOR(
      gradI.dataType(), upsampling3dBPCudaLauncher,
      (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), gradO.specialBuffer(),
       gradO.specialShapeInfo(), gradI.specialBuffer(), gradI.specialShapeInfo(), isNCDHW),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&gradI}, {&gradO});

  manager.synchronize();
}

}  // namespace ops
}  // namespace sd
