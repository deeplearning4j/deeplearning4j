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
//  @author raver119@gmail.com
//

#ifndef LIBND4J_SPECIAL_KERNELS_H
#define LIBND4J_SPECIAL_KERNELS_H

#include <helpers/shape.h>

#include <helpers/TAD.h>
#include <types/types.h>
#include <system/dll.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <helpers/DebugHelper.h>

namespace sd {

    template <typename T>
    _CUDA_H void fillIsMaxGeneric(dim3 &launchDims, cudaStream_t *stream, void *dx, const Nd4jLong *xShapeInfo, Nd4jLong length, long idx);

    template <typename T>
    _CUDA_H void fillDimensionalIsMaxGeneric(dim3 &launchDims, cudaStream_t *stream, const void *dX, void *dZ, const Nd4jLong *zShapeInfo, const Nd4jLong *tadOnlyShapeInfo, int *dimension, int dimensionLength, const Nd4jLong *tadOffsets);

    template<typename T>
    _CUDA_H void convertToHalfGeneric(dim3 &launchDims, cudaStream_t *stream, void *dx, Nd4jLong n, half *dz);

    template<typename T>
    _CUDA_H void tearKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong const* xShapeInfo, Nd4jPointer *targets,
                      Nd4jLong const* zShapeInfo, Nd4jLong const* tadShapeInfo, Nd4jLong const* tadOffsets);

    template<typename T>
    _CUDA_H void shuffleKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void **vdX, Nd4jLong **xShapeInfo, void **vdZ, int N,
                         int *shuffleMap, Nd4jLong** tadOnlyShapeInfo, Nd4jLong** tadOffsets);

    template<typename T>
    _CUDA_H void convertHalfsToGeneric(dim3 &launchDims, cudaStream_t *stream, half *dx, Nd4jLong n, void *dz);

    template<typename T>
    _CUDA_H void concatKernelVStackGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, Nd4jPointer *data,
                                            Nd4jPointer *inputShapeInfos, void *vz, Nd4jLong const* zShapeInfo);

    template<typename T>
    _CUDA_H void concatKernelScalarGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, Nd4jPointer *data, void *vresult);

    template<typename T>
    _CUDA_H void concatKernelHStackGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, Nd4jPointer *data,
                                            Nd4jPointer *inputShapeInfos, void *vresult, Nd4jLong const* resultShapeInfo);

    template<typename T>
    _CUDA_H void concatKernelGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, Nd4jPointer *data,
                        Nd4jPointer *inputShapeInfos, void *vresult, Nd4jLong const* resultShapeInfo,
                        Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers, Nd4jLong const* zTadShape, Nd4jLong const* zOffsets);

    template<typename T>
    _CUDA_H void pullRowsKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, void *vz, Nd4jLong n, Nd4jLong *indexes,
                          Nd4jLong const* tadShapeInfo, Nd4jLong const* tadOffsets, Nd4jLong const* zTadShapeInfo, Nd4jLong const* zTadOffsets);

    template<typename T>
    _CUDA_H void averagingKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void **vdx, void *vdz, int n, Nd4jLong length, bool propagate);

    template<typename T>
    _CUDA_H void accumulateKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void **vx, void *vz, int n, const Nd4jLong length);

    template<typename T>
    _CUDA_H void flattenKernelGeneric(dim3& launchDims, cudaStream_t *stream, Nd4jPointer *extraPointers, int dOffset, char order, void *vz, Nd4jLong *zShapeInfo, void *vy, Nd4jLong *yShapeInfo);

    template <typename T>
    _CUDA_H void tileKernelH(void const* inputBuffer, Nd4jLong const* inputShape, void* outputBuffer, Nd4jLong const* outputShape, Nd4jLong resultLength, cudaStream_t *stream);
    template <typename X, typename Y>
    _CUDA_H void tileKernelHH(void const* inputBuffer, Nd4jLong const* inputShape, void* outputBuffer, Nd4jLong const* outputShape, Nd4jLong resultLength, Nd4jLong ews, cudaStream_t *stream);

    class NDArray;
    template <typename T>
    _CUDA_H void setDiagonalValueUpper(void* buffer, Nd4jLong const* shape, NDArray const& value, int diagonal, Nd4jLong rows, Nd4jLong cols, cudaStream_t& stream);

    template <typename T>
    _CUDA_H void setDiagonalValueLower(void* buffer, Nd4jLong const* shape, NDArray const& value, int diagonal, Nd4jLong rows, Nd4jLong cols, cudaStream_t& stream);

    template <typename T>
    _CUDA_H void templatedSwapUnsafe(void* theFirstBuffer, Nd4jLong const* theFirstShape, void* theSecondBuffer, Nd4jLong const* theSecondShape, cudaStream_t* theStream);

}

#endif