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
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
SD_DEVICE void concatKernel(
   int numArrays,
   Pointer* data,
   Pointer* inputShapeInfos,
   void* vz,
   LongType* resultShapeInfo,
   Pointer* tadPointers,
   Pointer* offsetPointers,
   LongType* zTadShape,
   LongType* zOffsets) {

 const int tid = threadIdx.x + blockIdx.x * blockDim.x;

 // Shared variables for shape data
 __shared__ int zRank;
 __shared__ const LongType* zShape;
 __shared__ const LongType* zStride;
 __shared__ int zTadRank;
 __shared__ const LongType* zTadShapeOf;
 __shared__ const LongType* zTadStride;

 if (threadIdx.x == 0) {
   zRank = shape::rank(resultShapeInfo);
   zShape = shape::shapeOf(resultShapeInfo);
   zStride = shape::stride(resultShapeInfo);
   zTadRank = shape::rank(zTadShape);
   zTadShapeOf = shape::shapeOf(zTadShape);
   zTadStride = shape::stride(zTadShape);
 }
 __syncthreads();

 auto result = reinterpret_cast<T*>(vz);
 auto dataT = reinterpret_cast<T**>(data);
 auto shapeInfoPtrs = reinterpret_cast<LongType**>(inputShapeInfos);
 auto tadShapes = reinterpret_cast<LongType**>(tadPointers);
 auto tadOffsets = reinterpret_cast<LongType**>(offsetPointers);

 __shared__ bool _vec;
 __shared__ int baseIdx;
 __shared__ int arrOffset;
 __shared__ int yLength;
 __shared__ char yOrder;
 __shared__ int numTads;
 __shared__ char zOrder;

 if (threadIdx.x == 0) {
   zOrder = shape::order(resultShapeInfo);
   _vec = shape::isVector(resultShapeInfo);
 }
 __syncthreads();

 const int zLength = shape::length(resultShapeInfo);

 // Special case for when result is a vector
 if (_vec) {
   for (int r = blockIdx.x; r < numArrays; r += gridDim.x) {
     auto currShapeInfo = shapeInfoPtrs[r];
     if (shape::isVector(currShapeInfo) || shape::order(currShapeInfo) == zOrder) {
       if (threadIdx.x == 0) {
         yLength = shape::length(currShapeInfo);
         baseIdx = 0;
         for (int f = 0; f < r; f++) {
           baseIdx += shape::length(shapeInfoPtrs[f]);
         }
       }
       __syncthreads();

       for (int i = threadIdx.x; i < yLength && baseIdx + i < zLength; i += blockDim.x) {
         result[baseIdx + i] = dataT[r][i];
       }
       __syncthreads();
     }
   }
   return;
 }

 // For all non-vector results
 for (int r = 0; r < numArrays; r++) {
   auto currShape = shapeInfoPtrs[r];
   auto currData = dataT[r];
   auto currTad = tadShapes[r];
   auto currOffsets = tadOffsets[r];

   // Shared variables for current TAD shape data
   __shared__ int currTadRank;
   __shared__ const LongType* currTadShape;
   __shared__ const LongType* currTadStride;

   if (threadIdx.x == 0) {
     currTadRank = shape::rank(currTad);
     currTadShape = shape::shapeOf(currTad);
     currTadStride = shape::stride(currTad);
   }
   __syncthreads();

   if (threadIdx.x == 0) {
     yLength = shape::length(currTad);
     yOrder = shape::order(currTad);
     numTads = shape::length(currShape) / yLength;

     arrOffset = 0;
     for (int f = 0; f < r; f++) {
       arrOffset += shape::length(tadShapes[f]);
     }
   }
   __syncthreads();

   // Case for single element
   if (yLength == 1 && _vec) {
     for (LongType j = tid; j < numTads; j += blockDim.x * gridDim.x) {
       LongType inputOffset = currOffsets[j];
       LongType resultOffset = zOffsets[j];

       T* dataTAD = currData + inputOffset;
       T* resultTAD = result + resultOffset;

       LongType sub[SD_MAX_RANK];
       INDEX2COORDS(arrOffset, zTadRank, zTadShapeOf, sub);

       LongType baseOffset;
       COORDS2INDEX(zTadRank, zTadStride, sub, baseOffset);
       resultTAD += baseOffset;

       INDEX2COORDS(0, currTadRank, currTadShape, sub);

       LongType yOffset;
       COORDS2INDEX(currTadRank, currTadStride, sub, yOffset);

       COORDS2INDEX(zTadRank, zTadStride, sub, resultOffset);
       resultTAD[resultOffset] = dataTAD[yOffset];
     }
   }
   else {
     for (LongType j = blockIdx.x; j < numTads; j += gridDim.x) {
       const LongType inputOffset = currOffsets[j];
       const LongType resultOffset = zOffsets[j];

       T* dataTAD = currData + inputOffset;
       T* resultTAD = result + resultOffset;

       LongType sub[SD_MAX_RANK];
       INDEX2COORDS(arrOffset, zTadRank, zTadShapeOf, sub);

       LongType baseOffset;
       COORDS2INDEX(zTadRank, zTadStride, sub, baseOffset);
       resultTAD += baseOffset;

       if (zOrder == yOrder) {
         for (int i = threadIdx.x; i < yLength; i += blockDim.x) {
           resultTAD[i] = dataTAD[i];
         }
       }
       else {
         if (shape::order(resultShapeInfo) == shape::order(currTad)) {
           if (threadIdx.x == 0) {
             baseIdx = 0;
             for (int f = 0; f < r; f++) {
               baseIdx += shape::length(shapeInfoPtrs[f]);
             }
           }
           __syncthreads();

           if (numTads == 1) {
             for (int k = threadIdx.x; k < yLength; k += blockDim.x) {
               resultTAD[baseIdx + k] = dataTAD[k];
             }
           }
           else {
             LongType yIdx[SD_MAX_RANK];
             for (LongType i = threadIdx.x; i < yLength; i += blockDim.x) {
               INDEX2COORDS(i, currTadRank, currTadShape, yIdx);

               LongType yOffset;
               COORDS2INDEX(currTadRank, currTadShape, yIdx, yOffset);

               resultTAD[baseIdx + i] = dataTAD[yOffset];
             }
           }
           __syncthreads();
         }
         else {
           LongType zIdx[SD_MAX_RANK];
           LongType yIdx[SD_MAX_RANK];

           for (LongType i = threadIdx.x; i < yLength; i += blockDim.x) {
             INDEX2COORDS(i, currTadRank, currTadShape, yIdx);
             INDEX2COORDS(i, zTadRank, zTadShapeOf, zIdx);

             LongType yOffset;
             COORDS2INDEX(currTadRank, currTadShape, yIdx, yOffset);

             LongType rOffset;
             COORDS2INDEX(zTadRank, zTadShapeOf, zIdx, rOffset);
             resultTAD[rOffset] = dataTAD[yOffset];
           }
         }
       }
       __syncthreads();
     }
   }
   __syncthreads();
 }
}

template <typename T>
SD_KERNEL void execConcatKernel(
   int numArrays,
   Pointer* data,
   Pointer* inputShapeInfos,
   void* vz,
   LongType* zShapeInfo,
   Pointer* tadPointers,
   Pointer* offsetPointers,
   LongType* zTadShape,
   LongType* zOffsets) {

 concatKernel<T>(
     numArrays,
     data,
     inputShapeInfos,
     vz,
     zShapeInfo,
     tadPointers,
     offsetPointers,
     zTadShape,
     zOffsets);
}

template <typename T>
SD_HOST void concatKernelGeneric(
   dim3 &launchDims,
   cudaStream_t *stream,
   int numArrays,
   Pointer* data,
   Pointer* inputShapeInfos,
   void* vz,
   LongType* zShapeInfo,
   Pointer* tadPointers,
   Pointer* offsetPointers,
   LongType* zTadShape,
   LongType* zOffsets) {

 execConcatKernel<T>
     <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
         numArrays,
         data,
         inputShapeInfos,
         vz,
         zShapeInfo,
         tadPointers,
         offsetPointers,
         zTadShape,
         zOffsets);

 DebugHelper::checkErrorCode(stream, "concatGenericLegacy(...) failed");
}

BUILD_SINGLE_TEMPLATE(
   template void concatKernelGeneric,
   (dim3 &launchDims,
    cudaStream_t *stream,
    int numArrays,
    sd::Pointer* data,
    sd::Pointer* inputShapeInfos,
    void* vz,
    sd::LongType* zShapeInfo,
    sd::Pointer* tadPointers,
    sd::Pointer* offsetPointers,
    sd::LongType* zTadShape,
    sd::LongType* zOffsets),
   SD_COMMON_TYPES);

}  // namespace sd