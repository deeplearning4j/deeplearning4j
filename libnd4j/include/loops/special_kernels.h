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

__host__ void fillIsMaxGeneric(dim3& launchDims, cudaStream_t *stream, bool* dx, long length, long idx);

__host__ void fillDimensionalIsMaxGeneric(dim3& launchDims, cudaStream_t *stream, void *dX, bool *dZ, Nd4jLong *zShapeInfo, Nd4jLong *tadOnlyShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOffsets);

template <typename T>
__host__ void convertToHalfGeneric(dim3& launchDims, cudaStream_t *stream, void *dx, Nd4jLong n, half *dz);

template<typename T>
__host__ void tearKernelGeneric(dim3& launchDims, cudaStream_t *stream, void *vx, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

template<typename T>
__host__ void shuffleKernelGeneric(dim3& launchDims, cudaStream_t *stream, void **vdX, Nd4jLong **xShapeInfo,  void **vdZ, int N, int *shuffleMap, Nd4jLong **tadOnlyShapeInfo, Nd4jLong **tadOffsets);

template <typename T>
__host__ void convertHalfsToGeneric(dim3& launchDims, cudaStream_t *stream, half *dx, Nd4jLong n, void *dz);

template <typename T>
__host__ void concatKernelVStackGeneric(dim3& launchDims, cudaStream_t *stream, int numArrays, Nd4jPointer *data, Nd4jPointer *inputShapeInfos, void *vz, Nd4jLong *zShapeInfo);

template <typename T>
__host__ void concatKernelScalarGeneric(dim3& launchDims, cudaStream_t *stream, int numArrays, Nd4jPointer *data, void *vresult);

template <typename T>
__host__ void concatKernelHStackGeneric(dim3& launchDims, cudaStream_t *stream, int numArrays, Nd4jPointer *data, Nd4jPointer *inputShapeInfos, void *vresult, Nd4jLong *resultShapeInfo);

template <typename T>
__host__ void concatKernelGeneric(dim3& launchDims, cudaStream_t *stream, int dimension, int numArrays, Nd4jPointer *data, Nd4jPointer *inputShapeInfos, void *vresult, Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers, Nd4jLong *zTadShape, Nd4jLong *zOffsets);

template <typename T>
__host__ void pullRowsKernelGeneric(dim3& launchDims, cudaStream_t *stream, void *vx, void *vz, Nd4jLong n, Nd4jLong *indexes, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *zTadShapeInfo, Nd4jLong *zTadOffsets);

template <typename T>
__host__ void averagingKernelGeneric(dim3& launchDims, cudaStream_t *stream, void **vdx, void *vdz, int n, Nd4jLong length, bool propagate);

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
__host__ void accumulateKernelGeneric(dim3& launchDims, cudaStream_t *stream, void **vx, void *vz, int n, const Nd4jLong length);




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