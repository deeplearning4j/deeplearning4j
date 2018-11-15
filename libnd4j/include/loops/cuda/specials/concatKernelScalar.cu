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
// @author raver119@gmail.com
// @author Yurii Shyrma, created on 15.11.2018
//

#include <loops/special_kernels.h>

///////////////////////////////////////////////////////////////////////
template <typename T>
__device__ void concatKernelScalar(int dimension,
									int numArrays,
									Nd4jPointer *data, Nd4jPointer *inputShapeInfos,
									void *vz, Nd4jLong *zShapeInfo, 
									Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    auto z = static_cast<T*>(vz);
    Nd4jLong tid = blockIdx.x * blockDim.x + threadIdx.x;
    T **input = (T **) data;

    for (int i = tid; i < numArrays; i += blockDim.x * gridDim.x)
		z[i] = input[i][0];
}

///////////////////////////////////////////////////////////////////////
template <typename T>
__global__ void execConcatKernelScalar(int dimension,
									int numArrays,
									Nd4jPointer *data, Nd4jPointer *inputShapeInfos,
									void *vz, Nd4jLong *zShapeInfo, 
									Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

	concatKernelScalar<T>(dimension, numArrays, data, inputShapeInfos, vz, zShapeInfo, tadPointers, offsetPointers);
}

///////////////////////////////////////////////////////////////////////
template <typename T>
__host__ void concatKernelScalarGeneric(dim3& launchDims, Nd4jPointer* extraPointers,
									int dimension,
									int numArrays,
									Nd4jPointer *data, Nd4jPointer *inputShapeInfos,
									void *vz, Nd4jLong *zShapeInfo, 
									Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

	cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	concatKernelScalar<T><<<launchDims.x, launchDims.y, launchDims.z, stream>>>(dimension, numArrays, data, inputShapeInfos, vz, zShapeInfo, tadPointers, offsetPointers);
}