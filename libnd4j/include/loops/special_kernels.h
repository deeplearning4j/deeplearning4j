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
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <helpers/DebugHelper.h>
#include <helpers/TAD.h>
#include <helpers/shape.h>

namespace sd {

template <typename T>
SD_HOST void fillIsMaxGeneric(dim3 &launchDims, cudaStream_t *stream, void *dx, const LongType *xShapeInfo,
                              LongType length, long idx);

template <typename T>
SD_HOST void fillDimensionalIsMaxGeneric(dim3 &launchDims, cudaStream_t *stream, const void *dX, void *dZ,
                                         const LongType *zShapeInfo, const LongType *tadOnlyShapeInfo,
                                         LongType *dimension, LongType dimensionLength, const LongType *tadOffsets);

template <typename T>
SD_HOST void convertToHalfGeneric(dim3 &launchDims, cudaStream_t *stream, void *dx, LongType n, half *dz);

template <typename T>
SD_HOST void tearKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, LongType const *xShapeInfo,
                               Pointer *targets, LongType const *zShapeInfo, LongType const *tadShapeInfo,
                               LongType const *tadOffsets);

template <typename T>
SD_HOST void shuffleKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void **vdX, LongType **xShapeInfo,
                                  void **vdZ, int N, int *shuffleMap, LongType **tadOnlyShapeInfo, LongType **tadOffsets);

template <typename T>
SD_HOST void convertHalfsToGeneric(dim3 &launchDims, cudaStream_t *stream, half *dx, LongType n, void *dz);

template <typename T>
SD_HOST void concatKernelVStackGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, Pointer *data,
                                       Pointer *inputShapeInfos, void *vz, LongType const *zShapeInfo);

template <typename T>
SD_HOST void concatKernelScalarGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, Pointer *data,
                                       void *vresult);

template <typename T>
SD_HOST void concatKernelHStackGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, Pointer *data,
                                       Pointer *inputShapeInfos, void *vresult, LongType const *resultShapeInfo);

template <typename T>
SD_HOST void concatKernelGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, Pointer *data,
                                 Pointer *inputShapeInfos, void *vresult, LongType const *resultShapeInfo,
                                 Pointer *tadPointers, Pointer *offsetPointers, LongType const *zTadShape,
                                 LongType const *zOffsets);

template <typename T>
SD_HOST void pullRowsKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, void *vz, LongType n,
                                   LongType *indexes, LongType const *tadShapeInfo, LongType const *tadOffsets,
                                   LongType const *zTadShapeInfo, LongType const *zTadOffsets);

template <typename T>
SD_HOST void averagingKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void **vdx, void *vdz, int n,
                                    LongType length, bool propagate);

template <typename T>
SD_HOST void accumulateKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void **vx, void *vz, int n,
                                     const LongType length);

template <typename T>
SD_HOST void flattenKernelGeneric(dim3 &launchDims, cudaStream_t *stream, Pointer *extraPointers, int dOffset,
                                  char order, void *vz, LongType *zShapeInfo, void *vy, LongType *yShapeInfo);

template <typename T>
SD_HOST void tileKernelH(void const *inputBuffer, LongType const *inputShape, void *outputBuffer,
                         LongType const *outputShape, LongType resultLength, cudaStream_t *stream);
template <typename X, typename Y>
SD_HOST void tileKernelHH(void const *inputBuffer, LongType const *inputShape, void *outputBuffer,
                          LongType const *outputShape, LongType resultLength, LongType ews,
                          cudaStream_t *stream);

class NDArray;
template <typename T>
SD_HOST void setDiagonalValueUpper(void *buffer, LongType const *shape, NDArray const &value, int diagonal,
                                   LongType rows, LongType cols, cudaStream_t &stream);

template <typename T>
SD_HOST void setDiagonalValueLower(void *buffer, LongType const *shape, NDArray const &value, int diagonal,
                                   LongType rows, LongType cols, cudaStream_t &stream);

template <typename T>
SD_HOST void templatedSwapUnsafe(void *theFirstBuffer, LongType const *theFirstShape, void *theSecondBuffer,
                                 LongType const *theSecondShape, cudaStream_t *theStream);

}  // namespace sd

#endif
