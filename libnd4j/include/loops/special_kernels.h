/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_SPECIAL_KERNELS_H
#define LIBND4J_SPECIAL_KERNELS_H

#include <helpers/shape.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>

template <typename T>
__device__ void fillIsMaxGeneric(T *dx, long length, long idx);


template <typename T>
__device__ void fillDimensionalIsMaxGeneric(T *dX, Nd4jLong *xShapeInfo, T *dZ, Nd4jLong *zShapeInfo, Nd4jLong *tadOnlyShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOffsets);


template <typename T>
__device__ void concatKernelGeneric(int dimension, int numArrays, Nd4jPointer *data, Nd4jPointer *inputShapeInfos, void *vresult, Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers, Nd4jLong *zTadShape, Nd4jLong *zOffsets);


template <typename T>
__device__ void concatKernelScalarGeneric(int dimension, int numArrays, Nd4jPointer *data, Nd4jPointer *inputShapeInfos, void *vresult, Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers);


template <typename T>
__device__ void concatKernelHStackGeneric(int dimension, int numArrays, Nd4jPointer *data, Nd4jPointer *inputShapeInfos, void *vresult, Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers);


template <typename T>
__device__ void concatKernelVStackGeneric(int dimension, int numArrays, Nd4jPointer *data, Nd4jPointer *inputShapeInfos, void *vresult, Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers);


template <typename T>
__device__ void pullRowsKernelGeneric(void *vx, Nd4jLong *xShapeInfo, void *vz, Nd4jLong *zShapeInfo, Nd4jLong n, Nd4jLong *indexes, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *zTadShapeInfo, Nd4jLong *zTadOffsets);


template <typename T>
__device__ void convertToHalfGeneric(T *dx, Nd4jLong n, half *dz);


template <typename T>
__device__ void convertHalfsToGeneric(half *dx, Nd4jLong n, T *dz);

/**
 * This kernel accumulates X arrays, and stores z into Z
 *
 * @tparam T
 * @param x
 * @param z
 * @param n
 * @param length
 */
template<typename T>
__device__ void accumulateKernelGeneric(void **vx, void *vz, int n, const Nd4jLong length);


template <typename T>
__device__ void averagingKernelGeneric(void **vdx, void *vdz, int n, Nd4jLong length, bool propagate);


template<typename T>
__device__ void tearKernelGeneric(void *vx, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);


template<typename T>
__device__ void shuffleKernelGeneric(void **vdX, Nd4jLong **xShapeInfo, void **vdZ, Nd4jLong **zShapeInfo, int N, int *shuffleMap, Nd4jLong **tadOnlyShapeInfo, Nd4jLong **tadOffsets);


// extern "C" __global__ void prepareDimensionalShapeBuffer(Nd4jLong *xShapeInfoBuffer, float *extraParams, Nd4jLong *zShapeInfo) {
//     // extraParams[0] - number of dimensions
//     // extraParams[1] - dimension
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid > 0)
//         return;

//     int targetDimension = (int) extraParams[1];
//     //printf("Target dimension: [%i]\n", targetDimension);

//     int targetWidth = shape::shapeOf(xShapeInfoBuffer)[targetDimension];
//     //printf("Target rank: [%i]\n", targetWidth);
// }

#endif