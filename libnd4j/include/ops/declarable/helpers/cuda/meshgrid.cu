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
// @author raver119@gmail.com
//

#include <array/ResultSet.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/meshgrid.h>

#include <numeric>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static SD_DEVICE void assign_(void *vx, LongType *xShapeInfo, void *vz, LongType *zShapeInfo) {
  auto x = reinterpret_cast<T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto step = blockDim.x * gridDim.x;

  __shared__ LongType length, rankX, rankZ;
  __shared__ const LongType *shapeX, *strideX, *shapeZ, *strideZ;

  if (threadIdx.x == 0) {
    length = shape::length(xShapeInfo);
    rankX = shape::rank(xShapeInfo);
    rankZ = shape::rank(zShapeInfo);
    shapeX = shape::shapeOf(xShapeInfo);
    strideX = shape::stride(xShapeInfo);
    shapeZ = shape::shapeOf(zShapeInfo);
    strideZ = shape::stride(zShapeInfo);
  }
  __syncthreads();

  LongType xCoords[SD_MAX_RANK];
  LongType zCoords[SD_MAX_RANK];

  for (LongType i = tid; i < length; i += step) {
    // Compute input coordinates and offset
    INDEX2COORDS(i, rankX, shapeX, xCoords);
    LongType xOffset;
    COORDS2INDEX(rankX, strideX, xCoords, xOffset);

    // Compute output coordinates and offset
    INDEX2COORDS(i, rankZ, shapeZ, zCoords);
    LongType zOffset;
    COORDS2INDEX(rankZ, strideZ, zCoords, zOffset);

    // Assign value from input to output
    z[zOffset] = x[xOffset];
  }
}


template <typename T>
static SD_KERNEL void meshgridKernel(int rank, void **outBuffers, LongType **tadShapes, LongType **tadOffsets,
                                     LongType *numTads, void **inBuffers, LongType **inShapes) {
  // for all arrays
  for (int i = blockIdx.x; i < rank; i += gridDim.x) {
    // for all tads in this array
    for (LongType j = 0; j < numTads[i]; j++) {
      assign_<T>(inBuffers[i], inShapes[i], reinterpret_cast<T *>(outBuffers[i]) + tadOffsets[i][j], tadShapes[i]);
    }
    __syncthreads();
  }
}

template <typename T>
static void meshgrid_(LaunchContext *context, const std::vector<NDArray *> &inArrs,
                      const std::vector<NDArray *> &outArrs, const bool swapFirst2Dims) {
  const int rank = inArrs.size();
  int inIndices[SD_MAX_RANK];
  std::iota(inIndices, inIndices + rank, 0);
  if (swapFirst2Dims && rank > 1) {
    inIndices[0] = 1;
    inIndices[1] = 0;
  }

  PointersManager pm(context, "meshgrid");
  std::vector<const void *> hInBuffers(rank);
  std::vector<void *> hOutBuffers(rank);
  std::vector<const LongType *> hInShapes(rank);

  std::vector<const LongType *> hOutTadShapes(rank);
  std::vector<const LongType *> hOutTadOffsets(rank);

  std::vector<LongType> hNumTads(rank);

  for (int i = 0; i < rank; ++i) {
    hInBuffers[i] = inArrs[i]->specialBuffer();
    hInShapes[i] = inArrs[i]->specialShapeInfo();

    hOutBuffers[i] = outArrs[i]->specialBuffer();

    auto pack = ConstantTadHelper::getInstance().tadForDimensions(outArrs[i]->shapeInfo(), {inIndices[i]});
    hOutTadShapes[i] = pack->specialShapeInfo();
    hOutTadOffsets[i] = pack->specialOffsets();
    hNumTads[i] = pack->numberOfTads();

  }

  auto dInBuffers =
      reinterpret_cast<void **>(pm.replicatePointer(hInBuffers.data(), hInBuffers.size() * sizeof(void *)));
  auto dOutBuffers =
      reinterpret_cast<void **>(pm.replicatePointer(hOutBuffers.data(), hOutBuffers.size() * sizeof(void *)));

  auto dInShapes = reinterpret_cast<LongType **>(
      pm.replicatePointer(hInShapes.data(), hInShapes.size() * sizeof(LongType *)));
  auto dOutTadShapes = reinterpret_cast<LongType **>(
      pm.replicatePointer(hOutTadShapes.data(), hOutTadShapes.size() * sizeof(LongType *)));
  auto dOutTadOffsets = reinterpret_cast<LongType **>(
      pm.replicatePointer(hOutTadOffsets.data(), hOutTadOffsets.size() * sizeof(LongType *)));

  auto dNumTads =
      reinterpret_cast<LongType *>(pm.replicatePointer(hNumTads.data(), hNumTads.size() * sizeof(LongType)));

  dim3 launchDims = getLaunchDims("meshgrid");
  meshgridKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *context->getCudaStream()>>>(rank, dOutBuffers, dOutTadShapes, dOutTadOffsets,
                                                                   dNumTads, dInBuffers, dInShapes);
  sd::DebugHelper::checkErrorCode(context->getCudaStream(), "meshgridKernel failed");

  pm.synchronize();
}

//////////////////////////////////////////////////////////////////////////
void meshgrid(LaunchContext *context, const std::vector<NDArray *> &inArrs, const std::vector<NDArray *> &outArrs,
              const bool swapFirst2Dims) {
  BUILD_SINGLE_SELECTOR(inArrs.at(0)->dataType(), meshgrid_, (context, inArrs, outArrs, swapFirst2Dims),
                        SD_NUMERIC_TYPES);

  for (auto v : outArrs) v->tickWriteDevice();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
