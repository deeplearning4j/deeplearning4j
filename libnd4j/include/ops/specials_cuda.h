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

#ifndef PROJECT_SPECIALS_CUDA_H
#define PROJECT_SPECIALS_CUDA_H

#include <helpers/shape.h>
#include <helpers/DebugHelper.h>

#ifdef __CUDACC__

////////////////////////////////////////////////////////////////////////
template<typename T>
__host__ void bitonicSortStepGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong const* xShapeInfo, int j, int k, int length, bool descending);

////////////////////////////////////////////////////////////////////////
template<typename T>
__host__ void bitonicArbitraryStepGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong const* xShapeInfo, int window, int length,  int reverse, bool descending);

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
__host__ void bitonicSortStepGenericKey(dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong const* xShapeInfo, void *vy, Nd4jLong const* yShapeInfo, int j, int k, int length, bool descending);

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
__host__ void bitonicArbitraryStepGenericKey(dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong const* xShapeInfo, void *vy, Nd4jLong const* yShapeInfo, int window, int length,  int reverse, bool descending);

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
__host__ void bitonicSortStepGenericValue(dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong const* xShapeInfo, void *vy, Nd4jLong const* yShapeInfo, int j, int k, int length, bool descending);

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
__host__ void bitonicArbitraryStepGenericValue(dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong const* xShapeInfo, void *vy, Nd4jLong const* yShapeInfo, int window, int length,  int reverse, bool descending);



////////////////////////////////////////////////////////////////////////
template<typename T>
__host__ void oesTadGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong const* xShapeInfo,  int *dimension, int dimensionLength, Nd4jLong const* tadShapeInfo, Nd4jLong const* tadOffsets, bool descending);

template <typename X, typename Y>
__host__ void oesTadGenericKey(dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong const* xShapeInfo,  void *vy, Nd4jLong const* yShapeInfo, int *dimension, int dimensionLength, Nd4jLong const* tadShapeInfo, Nd4jLong const* tadOffsets, bool descending);

template <typename X, typename Y>
__host__ void oesTadGenericValue(dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong const* xShapeInfo,  void *vy, Nd4jLong const* yShapeInfo, int *dimension, int dimensionLength, Nd4jLong const* tadShapeInfo, Nd4jLong const* tadOffsets, bool descending);

////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ void printCudaGlobal(void* pointer, const int len) {

    for(int i = 0; i < len; ++i)
        printf("%f, ", (double)reinterpret_cast<T*>(pointer)[i] );
    printf("\n");
}

////////////////////////////////////////////////////////////////////////
template<typename T>
__device__ void printCudaDevice(void* pointer, const int len, const int tid = 0) {

    if(blockIdx.x * blockDim.x + threadIdx.x != tid) return;
    for(int i = 0; i < len; ++i)
        printf("%f, ", (double)reinterpret_cast<T*>(pointer)[i] );
    printf("\n");
}

////////////////////////////////////////////////////////////////////////
template<typename T>
__host__ void printCudaHost(void* pointer, const int len, cudaStream_t& stream) {

    void* ptr = malloc(sizeof(T)*len);

    cudaMemcpyAsync(ptr, pointer, sizeof(T)*len, cudaMemcpyDeviceToHost, stream);
    cudaError_t cudaResult = cudaStreamSynchronize(stream);
    if(cudaResult != 0)
        throw std::runtime_error("printCudaHost:: cudaStreamSynchronize failed!");

    for(int i = 0; i < len; ++i)
        printf("%f, ", (double)reinterpret_cast<T*>(ptr)[i]);
    printf("\n");

    free(ptr);
}


#endif

#endif //PROJECT_SPECIALS_CUDA_H
