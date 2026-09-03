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


#ifndef PROJECT_RANDOM_IMPL_GEN_H
#define PROJECT_RANDOM_IMPL_GEN_H

#include <helpers/DebugHelper.h>
#include <loops/random.h>
#include <ops/specials_cuda.h>
#include <system/common.h>
#include <system/op_boilerplate.h>
#include <types/types.h>


using namespace randomOps;

template <typename T, typename OpClass>
static SD_INLINE SD_DEVICE void randomSingleGeneric(sd::Pointer state, void* z, sd::LongType const* zShapeBuffer,
                                                   void* extraArguments) {
 functions::random::RandomFunction<T>::template execTransformCuda<OpClass>(state, z, zShapeBuffer, extraArguments);
}

template <typename T, typename OpClass>
static SD_INLINE SD_DEVICE void randomDoubleGeneric(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer,
                                                   void* z, sd::LongType const* zShapeBuffer, void* extraArguments) {
 functions::random::RandomFunction<T>::template execTransformCuda<OpClass>(state, x, xShapeBuffer, z, zShapeBuffer,
                                                                           extraArguments);
}

template <typename T, typename OpClass>
static SD_INLINE SD_DEVICE void randomTripleGeneric(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer,
                                                   void const* y, sd::LongType const* yShapeBuffer, void* z,
                                                   sd::LongType const* zShapeBuffer, void* extraArguments) {
 functions::random::RandomFunction<T>::template execTransformCuda<OpClass>(state, x, xShapeBuffer, y, yShapeBuffer, z,
                                                                           zShapeBuffer, extraArguments);
}


namespace functions {
namespace random {

template <typename T>
template <typename OpClass>
void SD_DEVICE RandomFunction<T>::execTransformCuda(sd::Pointer state, void const* vx, sd::LongType const* xShapeBuffer,
                                                    void const* vy, sd::LongType const* yShapeBuffer, void* vz,
                                                    sd::LongType const* zShapeBuffer, void* vextraArguments) {
 auto x = reinterpret_cast<T const*>(vx);
 auto y = reinterpret_cast<T const*>(vy);
 auto z = reinterpret_cast<T*>(vz);
 auto extraArguments = reinterpret_cast<T*>(vextraArguments);

 if (OpClass::requiresSpecial) {
   OpClass::specialOpCuda(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
   return;
 } else {
   __shared__ sd::LongType length;
   __shared__ char xOrder;
   __shared__ char yOrder;
   __shared__ char zOrder;

   // Cache shape information for x buffer
   __shared__ sd::LongType xRank;
   __shared__ const sd::LongType* xShapePtr;
   __shared__ const sd::LongType* xStridePtr;

   // Cache shape information for y buffer
   __shared__ sd::LongType yRank;
   __shared__ const sd::LongType* yShapePtr;
   __shared__ const sd::LongType* yStridePtr;

   // Cache shape information for z buffer
   __shared__ sd::LongType zRank;
   __shared__ const sd::LongType* zShapePtr;
   __shared__ const sd::LongType* zStridePtr;

   __shared__ sd::graph::RandomGenerator* buffer;
   __shared__ unsigned char* cB;
   __shared__ unsigned char* dB;
   sd::graph::RandomGenerator* devBuffer;

   if (threadIdx.x == 0) {
     length = shape::length(zShapeBuffer);
     xOrder = shape::order(xShapeBuffer);
     yOrder = shape::order(yShapeBuffer);
     zOrder = shape::order(zShapeBuffer);

     // Cache all shape information in thread 0
     xRank = shape::rank(xShapeBuffer);
     xShapePtr = shape::shapeOf(xShapeBuffer);
     xStridePtr = shape::stride(xShapeBuffer);

     yRank = shape::rank(yShapeBuffer);
     yShapePtr = shape::shapeOf(yShapeBuffer);
     yStridePtr = shape::stride(yShapeBuffer);

     zRank = shape::rank(zShapeBuffer);
     zShapePtr = shape::shapeOf(zShapeBuffer);
     zStridePtr = shape::stride(zShapeBuffer);

     extern __shared__ unsigned char shmem[];
     buffer = (sd::graph::RandomGenerator*)shmem;
     cB = shmem;
     devBuffer = reinterpret_cast<sd::graph::RandomGenerator*>(state);
     dB = reinterpret_cast<unsigned char*>(state);
   }
   __syncthreads();

   // using this loop instead of memcpy
   for (int e = threadIdx.x; e < sizeof(sd::graph::RandomGenerator); e += blockDim.x)
     cB[e] = dB[e];

   __syncthreads();

   int tid = blockIdx.x * blockDim.x + threadIdx.x;

   for (sd::LongType i = tid; i < length; i += blockDim.x * gridDim.x) {
     sd::LongType xCoords[SD_MAX_RANK];
     sd::LongType yCoords[SD_MAX_RANK];
     sd::LongType zCoords[SD_MAX_RANK];
     sd::LongType xOffset;
     sd::LongType yOffset;
     sd::LongType zOffset;

     INDEX2COORDS(i, xRank, xShapePtr, xCoords);
     COORDS2INDEX(xRank, xStridePtr, xCoords, xOffset);
     INDEX2COORDS(i, yRank, yShapePtr, yCoords);
     COORDS2INDEX(yRank, yStridePtr, yCoords, yOffset);
     INDEX2COORDS(i, zRank, zShapePtr, zCoords);
     COORDS2INDEX(zRank, zStridePtr, zCoords, zOffset);

     z[zOffset] = OpClass::op(x[xOffset], y[yOffset], i, length, buffer, extraArguments);
   }
 }
}

template <typename T>
template <typename OpClass>
void SD_DEVICE RandomFunction<T>::execTransformCuda(sd::Pointer state, void const* vx, sd::LongType const* xShapeBuffer,
                                                   void* vz, sd::LongType const* zShapeBuffer, void* vextraArguments) {
 auto x = reinterpret_cast<T const*>(vx);
 auto z = reinterpret_cast<T*>(vz);
 auto extraArguments = reinterpret_cast<T*>(vextraArguments);

 __shared__ sd::LongType length;
 __shared__ char xOrder;
 __shared__ char zOrder;

 __shared__ sd::graph::RandomGenerator* buffer;
 __shared__ unsigned char* cB;
 __shared__ unsigned char* dB;
 __shared__ sd::graph::RandomGenerator* devBuffer;

 if (threadIdx.x == 0) {
   extern __shared__ unsigned char shmem[];
   buffer = (sd::graph::RandomGenerator*)shmem;
   cB = shmem;
   devBuffer = reinterpret_cast<sd::graph::RandomGenerator*>(state);
   dB = reinterpret_cast<unsigned char*>(state);

   length = shape::length(zShapeBuffer);
   xOrder = shape::order(xShapeBuffer);
   zOrder = shape::order(zShapeBuffer);
 }
 __syncthreads();

 // using this loop instead of memcpy
 for (int e = threadIdx.x; e < sizeof(sd::graph::RandomGenerator); e += blockDim.x) cB[e] = dB[e];

 __syncthreads();

 for (sd::LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x) {
   sd::LongType xCoords[SD_MAX_RANK];
   sd::LongType zCoords[SD_MAX_RANK];
   sd::LongType xOffset;
   sd::LongType zOffset;

   INDEX2COORDS(i, shape::rank(xShapeBuffer), shape::shapeOf(xShapeBuffer), xCoords);
   COORDS2INDEX(shape::rank(xShapeBuffer), shape::stride(xShapeBuffer), xCoords, xOffset);
   INDEX2COORDS(i, shape::rank(zShapeBuffer), shape::shapeOf(zShapeBuffer), zCoords);
   COORDS2INDEX(shape::rank(zShapeBuffer), shape::stride(zShapeBuffer), zCoords, zOffset);

   z[zOffset] = OpClass::op(x[xOffset], i, length, buffer, extraArguments);
 }
}
template <typename T>
template <typename OpClass>
void SD_DEVICE RandomFunction<T>::execTransformCuda(sd::Pointer state, void* vz, sd::LongType const* zShapeBuffer,
                                                   void* vextraArguments) {
 auto z = reinterpret_cast<T*>(vz);
 auto extraArguments = reinterpret_cast<T*>(vextraArguments);

 __shared__ sd::LongType length;
 __shared__ sd::graph::RandomGenerator* buffer;
 __shared__ unsigned char* cB;
 __shared__ unsigned char* dB;
 __shared__ sd::graph::RandomGenerator* devBuffer;

 if (threadIdx.x == 0) {
   extern __shared__ unsigned char shmem[];
   buffer = (sd::graph::RandomGenerator*)shmem;
   cB = shmem;
   devBuffer = reinterpret_cast<sd::graph::RandomGenerator*>(state);
   dB = reinterpret_cast<unsigned char*>(state);
   length = shape::length(zShapeBuffer);
 }
 __syncthreads();

 // using this loop instead of memcpy
 for (int e = threadIdx.x; e < sizeof(sd::graph::RandomGenerator); e += blockDim.x) cB[e] = dB[e];

 __syncthreads();

 int tid = blockIdx.x * blockDim.x + threadIdx.x;

 for (sd::LongType i = tid; i < length; i += blockDim.x * gridDim.x) {
   sd::LongType zCoords[SD_MAX_RANK];
   sd::LongType zOffset;

   INDEX2COORDS(i, shape::rank(zShapeBuffer), shape::shapeOf(zShapeBuffer), zCoords);
   COORDS2INDEX(shape::rank(zShapeBuffer), shape::stride(zShapeBuffer), zCoords, zOffset);

   z[zOffset] = OpClass::op(i, length, buffer, extraArguments);
 }
}


}  // namespace random
}  // namespace functions

#endif  // PROJECT_RANDOM_IMPL_GEN_H
