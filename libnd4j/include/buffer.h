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

/*
 * buffer.h
 *
 *  Created on: Dec 24, 2015
 *      Author: agibsonccc
 */

#ifndef BUFFER_H_
#define BUFFER_H_
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <dll.h>

#include <helper_string.h>
#include <helper_cuda.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <dll.h>

 //Question: Should the indexes here really be int? Isn't size_t or Nd4jLong more appropriate?
namespace nd4j {
	namespace buffer {
/**
 * Represents both a cpu and gpu
 * buffer - mainly used for testing
 */
		template<typename T>
		struct Buffer {
			int length = 0;
			int allocatedOnGpu = 0;
                        T *data = nullptr;
                        T *gData = nullptr;
			T one, two;
                public:
                        ~Buffer() {
                            delete []data;
                            delete []gData;
                        }

			void assign(T *val) {
				data = val;
			}

			T &operator=(T x) {
				one = x;
				return x;
			}

			class Proxy {
				Buffer<T> &a;
				int idx;
			public:
				Proxy(Buffer &a, int idx) :
						a(a), idx(idx) {
				}

				T &operator=(T x) {
					a.two = x;
					a.data[idx] = x;
					return a.data[idx];
				}
			};

		
			Proxy operator[](int index) {
				return Proxy(*this, index);
			}
		};

/**
 * Returns the size of the buffer
 * in bytes
 * @param buffer the buffer to get the size of
 * @return the size of the buffer in bytes
 */
		template<typename T>

#ifdef __CUDACC__
		__host__ __device__
#endif

		int bufferSize(Buffer<T> *buffer);

/**
 * Copies data to the gpu
 * @param buffer the buffer to copy
 */

#ifdef __CUDACC__
		template<typename T>
		__host__
		void copyDataToGpu(Buffer<T> **buffer, cudaStream_t stream);
#endif



/**
 * Copies data from the gpu
 * @param buffer the buffer to copy
 */

#ifdef __CUDACC__
		template<typename T>
		__host__
		void copyDataFromGpu(Buffer<T> **buffer, cudaStream_t stream);
#endif



/**
 * Allocate buffer of the given
 * length on the cpu and gpu.
 */
		template<typename T>
#ifdef __CUDACC__
		__host__
#endif
		void allocBuffer(Buffer<T> **buffer, int length);

/**
 * Frees the given buffer
 * (gpu and cpu
 */
		template<typename T>
#ifdef __CUDACC__
		__host__
#endif
		void freeBuffer(Buffer<T> **buffer);

/**
 * Creates a buffer
 * based on the data
 * and also synchronizes
 * the data on the gpu.
 */
		template<typename T>
#ifdef __CUDACC__
		__host__
#endif
		Buffer<T>
				*
				createBuffer(T *data, int length);

/**
 * Print the buffer on the host
 * @param buff
 */
		template<typename T>
#ifdef __CUDACC__
		__host__
#endif
		void printArr(Buffer<T> *buff);

/**
 *
 * @param buffer
 * @return
 */
		template<typename T>
#ifdef __CUDACC__
		__host__ __device__
#endif

		int bufferSize(Buffer<T> *buffer) {
			return sizeof(T) * buffer->length;
		}

#ifdef __CUDACC__
		/**
 *
 * @param buffer
 */
template<typename T>
__host__ void copyDataToGpu(Buffer <T> **buffer, cudaStream_t stream) {
	Buffer <T> *bufferRef = *buffer;
	checkCudaErrors(cudaMemcpyAsync(bufferRef->gData, bufferRef->data, bufferSize(bufferRef), cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));
}

/**
 *
 * @param buffer
 */
template<typename T>
__host__ void copyDataFromGpu(Buffer <T> **buffer, cudaStream_t stream) {
	Buffer <T> *bufferRef = *buffer;
	int bufferTotalSize = bufferSize(bufferRef);
	checkCudaErrors(cudaMemcpyAsync(bufferRef->data, bufferRef->gData, bufferTotalSize, cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));
}
#endif

/**
 * Allocate buffer of the given
 * length on the cpu and gpu.
 */
		template<typename T>
#ifdef __CUDACC__
		__host__
#endif
		void allocBuffer(Buffer<T> **buffer, int length) {
			Buffer<T> *bufferRef = *buffer;
			bufferRef->length = length;
			bufferRef->data = reinterpret_cast<T *>(malloc(sizeof(T) * length));

			CHECK_ALLOC(bufferRef->data, "Failed to allocate new buffer");
#ifdef __CUDACC__
			checkCudaErrors(cudaMalloc(&bufferRef->gData, sizeof(T) * length));
#endif
		}

/**
 * Frees the given buffer
 * (gpu and cpu
 */
		template<typename T>
#ifdef __CUDACC__
		__host__
#endif

                void freeBuffer(Buffer<T> *buffer) {
#ifdef __CUDACC__
			if(buffer->gData != nullptr)
            checkCudaErrors(cudaFree(buffer->gData));
#endif

                        delete buffer;
		}

/**
 * Creates a buffer
 * based on the data
 * and also synchronizes
 * the data on the gpu.
 */
		template<typename T>
#ifdef __CUDACC__
		__host__
#endif
		Buffer<T> *createBuffer(T *data, int length) {
                        Buffer<T> *ret = new Buffer<T>;
                        T *buffData = new T[length];
			for(int i = 0; i < length; i++)
				buffData[i] = data[i];
			ret->data = buffData;
			ret->length = length;
			return ret;
		}



#ifdef __CUDACC__
		template<typename T>
		__host__
		Buffer<T> *createBuffer(T *data, int length, cudaStream_t stream) {
			Buffer<T> *ret = createBuffer(data, length);

			T *gData;
			T **gDataRef = &(gData);
			checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(gDataRef), sizeof(T) * length));
			ret->gData = gData;
			checkCudaErrors(cudaMemcpyAsync(ret->gData, ret->data, sizeof(T) * length, cudaMemcpyHostToDevice, stream));
			return ret;
		}
#endif
	}
}



#ifdef __CUDACC__
template<typename T>
__host__ void printArr(nd4j::buffer::Buffer <T> *buff) {
	for (int i = 0; i < buff->length; i++) {
		printf("Buffer[%d] was %f\n", i, buff->data[i]);
	}
}

#endif

#endif /* BUFFER_H_ */
