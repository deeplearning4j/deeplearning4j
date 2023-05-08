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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//

#include <array/NDArrayFactory.h>
#include <array/ResultSet.h>
#include <exceptions/cuda_exception.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <ops/declarable/helpers/transforms.h>

#include <numeric>

namespace sd {
namespace ops {
namespace helpers {
///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void scatterUpdateCuda(const int opCode, const int numOfInd, void* vx, const sd::LongType* xShapeInfo,
                                        const sd::LongType* xOffsets, void* vy, const sd::LongType* yShapeInfo,
                                        const sd::LongType* yOffsets, const LongType* indexes) {
  __shared__ T *x, *y;
  __shared__ sd::LongType arrLenX, arrLenY;

  for (int e = 0; e < numOfInd; e++) {
    const auto xIndex = indexes[e];
    const bool isOwner = xIndex < gridDim.x ? blockIdx.x == xIndex : blockIdx.x == xIndex % gridDim.x;

    if (!isOwner) continue;

    if (threadIdx.x == 0) {
      x = reinterpret_cast<T*>(vx) + xOffsets[xIndex];
      y = reinterpret_cast<T*>(vy) + yOffsets[e];
      arrLenX = shape::length(xShapeInfo);
      arrLenY = shape::length(yShapeInfo);
    }
    __syncthreads();

    if (arrLenX != arrLenY) return;

    for (sd::LongType i = threadIdx.x; i < arrLenX; i += blockDim.x) {
      const auto xOffset = shape::getIndexOffset(i, xShapeInfo);
      const auto yOffset = shape::getIndexOffset(i, yShapeInfo);

      switch (opCode) {
        case 0:
          x[xOffset] += y[yOffset];
          break;
        case 1:
          x[xOffset] -= y[yOffset];
          break;
        case 2:
          x[xOffset] *= y[yOffset];
          break;
        case 3:
          x[xOffset] /= y[yOffset];
          break;
        case 4:
          x[xOffset] = y[yOffset] - x[xOffset];
          break;
        case 5:
          x[xOffset] = y[yOffset] / x[xOffset];
          break;
        case 6:
          x[xOffset] = y[yOffset];
          break;
        default:
          continue;
      }
    }
    __syncthreads();
  }
}

template <typename T>
SD_HOST static void scatterUpdateCudaLauncher(const cudaStream_t* stream, const int opCode, const int numOfInd,
                                              void* vx, const sd::LongType* xShapeInfo, const sd::LongType* xOffsets,
                                              void* vy, const sd::LongType* yShapeInfo, const sd::LongType* yOffsets,
                                              const LongType* indexes) {
  scatterUpdateCuda<T><<<512, 256, SD_MAX_NUM_THREADS, *stream>>>(opCode, numOfInd, vx, xShapeInfo, xOffsets, vy,
                                                                  yShapeInfo, yOffsets, indexes);
}

//////////////////////////////////////////////////////////////////////////
void scatterUpdate(sd::LaunchContext* context, NDArray& input, NDArray& updates, const std::vector<sd::LongType>* intArgs) {
  const int opCode = (*intArgs)[0];
  const int numOfDims = (*intArgs)[1];
  const int numOfInd = (*intArgs)[2 + numOfDims];

  std::vector<sd::LongType> tadDimensions(numOfDims);
  for (int e = 2; e < 2 + numOfDims; e++) tadDimensions[e - 2] = (*intArgs)[e];

  auto packX = ConstantTadHelper::getInstance().tadForDimensions(input.shapeInfo(), tadDimensions);
  auto packY = ConstantTadHelper::getInstance().tadForDimensions(updates.shapeInfo(), tadDimensions);

  NDArray indices(const_cast<sd::LongType *>(intArgs->data()) + numOfDims + 3, 'c', {numOfInd}, sd::DataType::INT32, context);

  PointersManager manager(context, "scatterUpdate");

  NDArray::prepareSpecialUse({&input}, {&input, &updates, &indices});
  BUILD_SINGLE_SELECTOR(input.dataType(), scatterUpdateCudaLauncher,
                        (context->getCudaStream(), opCode, numOfInd, input.specialBuffer(), packX->platformShapeInfo(),
                         packX->platformOffsets(), updates.specialBuffer(), packY->platformShapeInfo(),
                         packY->platformOffsets(), reinterpret_cast<sd::LongType *>(indices.specialBuffer())),
                        SD_COMMON_TYPES);
  NDArray::registerSpecialUse({&input}, {&input, &updates, &indices});

  manager.synchronize();
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
