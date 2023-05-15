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
// @author raver119@gmail.com
//

#ifndef PROJECT_SPECIALSSD_HOST
#define PROJECT_SPECIALSSD_HOST
#include <helpers/DebugHelper.h>
#include <helpers/shape.h>

#ifdef __CUDACC__

////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void bitonicSortStepGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo,
                                    int j, int k, int length, bool descending);

////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void bitonicArbitraryStepGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx,
                                         sd::LongType const *xShapeInfo, int window, int length, int reverse,
                                         bool descending);

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST void bitonicSortStepGenericKey(dim3 &launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo,
                                       void *vy, sd::LongType const *yShapeInfo, int j, int k, int length,
                                       bool descending);

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST void bitonicArbitraryStepGenericKey(dim3 &launchDims, cudaStream_t *stream, void *vx,
                                            sd::LongType const *xShapeInfo, void *vy, sd::LongType const *yShapeInfo,
                                            int window, int length, int reverse, bool descending);

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST void bitonicSortStepGenericValue(dim3 &launchDims, cudaStream_t *stream, void *vx,
                                         sd::LongType const *xShapeInfo, void *vy, sd::LongType const *yShapeInfo,
                                         int j, int k, int length, bool descending);

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST void bitonicArbitraryStepGenericValue(dim3 &launchDims, cudaStream_t *stream, void *vx,
                                              sd::LongType const *xShapeInfo, void *vy, sd::LongType const *yShapeInfo,
                                              int window, int length, int reverse, bool descending);

////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void oesTadGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo,
                           sd::LongType *dimension, sd::LongType dimensionLength, sd::LongType const *tadShapeInfo,
                           sd::LongType const *tadOffsets, bool descending);

template <typename X, typename Y>
SD_HOST void oesTadGenericKey(dim3 &launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo,
                              void *vy, sd::LongType const *yShapeInfo, sd::LongType *dimension,
                              sd::LongType dimensionLength,
                              sd::LongType const *tadShapeInfo, sd::LongType const *tadOffsets, bool descending);

template <typename X, typename Y>
SD_HOST void oesTadGenericValue(dim3 &launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo,
                                void *vy, sd::LongType const *yShapeInfo, sd::LongType *dimension,
                                sd::LongType dimensionLength,
                                sd::LongType const *tadShapeInfo, sd::LongType const *tadOffsets, bool descending);

////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void printCudaGlobal(void *pointer, const int len) {
  for (int i = 0; i < len; ++i) printf("%f, ", (double)reinterpret_cast<T *>(pointer)[i]);
  printf("\n");
}

////////////////////////////////////////////////////////////////////////
template <typename T>
SD_DEVICE void printCudaDevice(void *pointer, const int len, const int tid = 0) {
  if (blockIdx.x * blockDim.x + threadIdx.x != tid) return;
  for (int i = 0; i < len; ++i) printf("%f, ", (double)reinterpret_cast<T *>(pointer)[i]);
  printf("\n");
}

////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void printCudaHost(void *pointer, const int len, cudaStream_t &stream) {
  void *ptr = malloc(sizeof(T) * len);

  cudaMemcpyAsync(ptr, pointer, sizeof(T) * len, cudaMemcpyDeviceToHost, stream);
  cudaError_t cudaResult = cudaStreamSynchronize(stream);
  if (cudaResult != 0) THROW_EXCEPTION("printCudaHost:: cudaStreamSynchronize failed!");

  for (int i = 0; i < len; ++i) printf("%f, ", (double)reinterpret_cast<T *>(ptr)[i]);
  printf("\n");

  free(ptr);
}

#endif

#endif  // PROJECT_SPECIALSSD_HOST
