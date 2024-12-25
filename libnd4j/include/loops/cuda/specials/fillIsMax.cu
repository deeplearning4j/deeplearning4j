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
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
 * the License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
// @author Yurii Shyrma, created on 15.11.2018
//
#include <loops/special_kernels.h>

    namespace sd {

  template <typename T>
  SD_KERNEL void execFillIsMax(
      void* vdZ,
      const LongType* xShapeInfo,
      LongType length,
      long idx) {

    auto dz = reinterpret_cast<T*>(vdZ);

    __shared__ int rank;
    __shared__ const sd::LongType* shapePtr;
    __shared__ const sd::LongType* stridePtr;

    if (threadIdx.x == 0) {
      rank      = shape::rank(xShapeInfo);
      shapePtr  = shape::shapeOf(xShapeInfo);
      stridePtr = shape::stride(xShapeInfo);
    }
    __syncthreads();

    const int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalThreads = blockDim.x * gridDim.x;

    for (LongType i = tid; i < length; i += totalThreads) {
      sd::LongType coords[SD_MAX_RANK];
      sd::LongType offset;

      INDEX2COORDS(i, rank, shapePtr, coords);
      COORDS2INDEX(rank, stridePtr, coords, offset);

      dz[offset] = (i == idx ? static_cast<T>(1) : static_cast<T>(0));
    }
  }

  template <typename T>
  SD_HOST void fillIsMaxGeneric(
      dim3 &launchDims,
      cudaStream_t *stream,
      void* dz,
      const LongType* zShapeInfo,
      LongType length,
      long idx) {

    execFillIsMax<T>
        <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
            dz,
            zShapeInfo,
            length,
            idx);

    DebugHelper::checkErrorCode(stream, "fillIsMax(...) failed");
  }

  BUILD_SINGLE_TEMPLATE(
      template void fillIsMaxGeneric,
      (dim3 & launchDims,
       cudaStream_t *stream,
       void *dz,
       const sd::LongType *zShapeInfo,
       sd::LongType length,
       long idx),
      SD_COMMON_TYPES);

}  // namespace sd
