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

#include <TAD.h>
#include <types/types.h>
#include <dll.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>

namespace nd4j {

    template <typename T>
    _CUDA_H void fillIsMaxGeneric(dim3 &launchDims, cudaStream_t *stream, void *dx, Nd4jLong length, long idx);

    template <typename T>
    _CUDA_H void fillDimensionalIsMaxGeneric(dim3 &launchDims, cudaStream_t *stream, void *dX, void *dZ, Nd4jLong *zShapeInfo, Nd4jLong *tadOnlyShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOffsets);

    template<typename T>
    _CUDA_H void convertToHalfGeneric(dim3 &launchDims, cudaStream_t *stream, void *dx, Nd4jLong n, half *dz);

    template<typename T>
    _CUDA_H void tearKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong *xShapeInfo, Nd4jPointer *targets,
                      Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

    template<typename T>
    _CUDA_H void shuffleKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void **vdX, Nd4jLong **xShapeInfo, void **vdZ, int N,
                         int *shuffleMap, Nd4jLong **tadOnlyShapeInfo, Nd4jLong **tadOffsets);

    template<typename T>
    _CUDA_H void convertHalfsToGeneric(dim3 &launchDims, cudaStream_t *stream, half *dx, Nd4jLong n, void *dz);

    template<typename T>
    _CUDA_H void concatKernelVStackGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, Nd4jPointer *data,
                                            Nd4jPointer *inputShapeInfos, void *vz, Nd4jLong *zShapeInfo);

    template<typename T>
    _CUDA_H void concatKernelScalarGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, Nd4jPointer *data, void *vresult);

    template<typename T>
    _CUDA_H void concatKernelHStackGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, Nd4jPointer *data,
                                            Nd4jPointer *inputShapeInfos, void *vresult, Nd4jLong *resultShapeInfo);

    template<typename T>
    _CUDA_H void concatKernelGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, Nd4jPointer *data,
                        Nd4jPointer *inputShapeInfos, void *vresult, Nd4jLong *resultShapeInfo,
                        Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers, Nd4jLong *zTadShape, Nd4jLong *zOffsets);

    template<typename T>
    _CUDA_H void pullRowsKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, void *vz, Nd4jLong n, Nd4jLong *indexes,
                          Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *zTadShapeInfo, Nd4jLong *zTadOffsets);

    template<typename T>
    _CUDA_H void averagingKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void **vdx, void *vdz, int n, Nd4jLong length, bool propagate);

    template<typename T>
    _CUDA_H void accumulateKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void **vx, void *vz, int n, const Nd4jLong length);

    template<typename T>
    _CUDA_H void flattenKernelGeneric(dim3& launchDims, cudaStream_t *stream, Nd4jPointer *extraPointers, int dOffset, char order, void *vz, Nd4jLong *zShapeInfo, void *vy, Nd4jLong *yShapeInfo);
}

#endif