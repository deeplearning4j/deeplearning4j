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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 07.03.2019
//

#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/gather.h>

#include <numeric>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"

namespace sd {
namespace ops {
namespace helpers {

template <typename X, typename Y>
SD_KERNEL static void gatherCudaLinearKernel(const void* vx, const LongType* xShapeInfo, const void* vy,
                                             const LongType* yShapeInfo, void* vz, const LongType* zShapeInfo) {
  __shared__ const X* x;
  __shared__ const Y* y;
  __shared__ X* z;
  __shared__ LongType xLen, yLen, zLen;

  if (threadIdx.x == 0) {
    x = reinterpret_cast<const X*>(vx);
    z = reinterpret_cast<X*>(vz);
    y = reinterpret_cast<const Y*>(vy);
    xLen = shape::length(xShapeInfo);
    yLen = shape::length(yShapeInfo);
    zLen = shape::length(zShapeInfo);
  }
  __syncthreads();
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (LongType j = start; j < zLen; j += step) {
    auto zIndex = shape::getIndexOffset(j, zShapeInfo);
    auto yIndex = shape::getIndexOffset(j, yShapeInfo);
    auto xIndex = shape::getIndexOffset(y[yIndex], xShapeInfo);
    z[zIndex] = x[xIndex];
  }
}

//////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_KERNEL static void gatherCuda(const int numOfSubArrs, const void* vx, const LongType* xShapeInfo,
                                 const LongType* xOffsets, const void* vy, const LongType* yShapeInfo, void* vz,
                                 const LongType* zShapeInfo, const LongType* zOffsets) {
  const Y* y = reinterpret_cast<const Y*>(vy);
  __shared__ const X* x;
  __shared__ X* z;

  const LongType len = shape::length(xShapeInfo);
  for (LongType i = blockIdx.x; i < numOfSubArrs; i += gridDim.x) {
    if (threadIdx.x == 0) {
      x = reinterpret_cast<const X*>(vx) + xOffsets[y[shape::getIndexOffset(i, yShapeInfo)]];
      z = reinterpret_cast<X*>(vz) + zOffsets[i];
      printf("gather len is %d processing block %d at i %d\n",len,blockIdx.x,i);
    }
    __syncthreads();


    for (LongType j = threadIdx.x; j < len; j += blockDim.x) {
      auto zIndex = shape::getIndexOffset(j, zShapeInfo);
      auto xIndex = shape::getIndexOffset(j, xShapeInfo);
      printf("Setting x index at %d and z index %d at j %d\n",xIndex,zIndex,j);
      z[zIndex] = x[xIndex];
    }
    __syncthreads();
  }
}

template <typename X, typename Y>
SD_HOST static void gatherCudaLinear(const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                                     const void* vy, const LongType* yShapeInfo, void* vz,
                                     const LongType* zShapeInfo) {
 //note gather linear and gather are different kernels
   dim3 gatherLinear = getLaunchDims("gather_linear");
  gatherCudaLinearKernel<X, Y><<<gatherLinear.x, gatherLinear.y, gatherLinear.z, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
  DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream),"gatherCudaLinearKernel failed");

}

//////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST static void gatherCudaLauncher(const cudaStream_t* stream, const int numOfSubArrs, const void* vx,
                                       const LongType* xShapeInfo, const LongType* xOffsets, const void* vy,
                                       const LongType* yShapeInfo, void* vz, const LongType* zShapeInfo,
                                       const LongType* zOffsets) {
  dim3 gatherLinear = getGatherLinear(numOfSubArrs);
  gatherCuda<X, Y><<<gatherLinear.y, gatherLinear.x, gatherLinear.z, *stream>>>(numOfSubArrs, vx, xShapeInfo, xOffsets, vy,
                                                                        yShapeInfo, vz, zShapeInfo, zOffsets);
  DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream),"gatherCudaLauncher failed");

}

//////////////////////////////////////////////////////////////////////
void gather(LaunchContext* context, const NDArray* input, const NDArray* indices, NDArray* output,
            const std::vector<LongType>& intArgs) {
  const LongType inputRank = input->rankOf();
  const LongType numOfIntArgs = intArgs.size();

  LongType axis = numOfIntArgs > 0 ? intArgs[0] : 0;
  if (axis < 0) axis += inputRank;

  if (indices == nullptr && numOfIntArgs == 2) {  // scalar case
    printf("case 1\n");
    output->assign((*input)(intArgs[1], {axis}));
  } else if (indices != nullptr && indices->isScalar()) {
    printf("case 2\n");
    if (input->rankOf() <= 1) {  // For scalar indices, rank 0 or 1 input: can't do tensor along dimension 0 as this is
                                 // whole array... instead, we want to get a scalar
      auto idx = indices->e<LongType>(0);
      auto scalarNDArray = input->e(idx);
      output->assign(scalarNDArray);
    } else {
      printf("case 3\n");
      NDArray inSubArr = (*input)(indices->e<LongType>(0), {axis});
      output->assign(inSubArr);
    }
  } else {
    printf("case 4\n");
    NDArray* pIndices = const_cast<NDArray*>(indices);
    if (indices == nullptr)
      pIndices =
          new NDArray(input->ordering(), {numOfIntArgs - 1}, std::vector<double>(intArgs.begin() + 1, intArgs.end()), INT64, input->getContext());

    std::vector<LongType> dimsOut(pIndices->rankOf());
    std::iota(dimsOut.begin(), dimsOut.end(), axis);  // fill with axis, axis+1, ... axis+pIndices->rankOf()-1

    const LongType numOfSubArrs = pIndices->lengthOf();

    LongType *outSubArrShapeInfo(nullptr), *inSubArrShapeInfo(nullptr), *outSubArrOffsets(nullptr),
        *inSubArrOffsets(nullptr);
    input->getSubArrShapeAndOffsets({axis}, inSubArrShapeInfo, inSubArrOffsets);
    output->getSubArrShapeAndOffsets(dimsOut, outSubArrShapeInfo, outSubArrOffsets);
    if (output->rankOf() > 1) {
      PointersManager manager(context, "gather");
      auto xShapeInfo = reinterpret_cast<LongType*>(
          manager.replicatePointer(inSubArrShapeInfo, shape::shapeInfoByteLength(inSubArrShapeInfo)));
      auto zShapeInfo = reinterpret_cast<LongType*>(
          manager.replicatePointer(outSubArrShapeInfo, shape::shapeInfoByteLength(outSubArrShapeInfo)));
      auto xOffsets = reinterpret_cast<LongType*>(manager.replicatePointer(
          inSubArrOffsets, (input->lengthOf() / shape::length(inSubArrShapeInfo)) * sizeof(LongType)));
      auto zOffsets = reinterpret_cast<LongType*>(manager.replicatePointer(
          outSubArrOffsets, (output->lengthOf() / shape::length(outSubArrShapeInfo)) * sizeof(LongType)));

      NDArray::prepareSpecialUse({output}, {input, pIndices});
      BUILD_DOUBLE_SELECTOR(
          input->dataType(), pIndices->dataType(), gatherCudaLauncher,
          (context->getCudaStream(), numOfSubArrs, input->specialBuffer(), xShapeInfo, xOffsets,
           pIndices->specialBuffer(), pIndices->specialShapeInfo(), output->specialBuffer(), zShapeInfo, zOffsets),
          SD_COMMON_TYPES, SD_INDEXING_TYPES);
      NDArray::registerSpecialUse({output}, {input, pIndices});
      manager.synchronize();
    } else {
      printf("case 5\n");
      NDArray::prepareSpecialUse({output}, {input, pIndices});
      BUILD_DOUBLE_SELECTOR(
          input->dataType(), pIndices->dataType(), gatherCudaLinear,
          (context->getCudaStream(), input->specialBuffer(), input->specialShapeInfo(), pIndices->specialBuffer(),
           pIndices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo()),
          SD_COMMON_TYPES, SD_INDEXING_TYPES);
      NDArray::registerSpecialUse({output}, {input, pIndices});
    }

    if (indices == nullptr) delete pIndices;
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
