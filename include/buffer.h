/*
 * buffer.h
 *
 *  Created on: Dec 24, 2015
 *      Author: agibsonccc
 */

#ifndef BUFFER_H_
#define BUFFER_H_

#include <helper_string.h>
#include <helper_cuda.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

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
			T *data = NULL;
			T *gData = NULL;
			T one, two;
		public:
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

		size_t bufferSize(Buffer<T> *buffer);

/**
 * Copies data to the gpu
 * @param buffer the buffer to copy
 */
		template<typename T>
#ifdef __CUDACC__
		__host__
#endif
		void copyDataToGpu(Buffer<T> **buffer);

/**
 * Copies data from the gpu
 * @param buffer the buffer to copy
 */
		template<typename T>
#ifdef __CUDACC__
		__host__
#endif
		void copyDataFromGpu(Buffer<T> **buffer);

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

		size_t bufferSize(Buffer<T> *buffer) {
			return sizeof(T) * buffer->length;
		}

#ifdef __CUDACC__
		/**
 *
 * @param buffer
 */
template<typename T>
__host__ void copyDataToGpu(Buffer <T> **buffer) {
	Buffer <T> *bufferRef = *buffer;
	checkCudaErrors(
			cudaMemcpy(bufferRef->gData, bufferRef->data, bufferSize(bufferRef), cudaMemcpyHostToDevice));
}

/**
 *
 * @param buffer
 */
template<typename T>
__host__ void copyDataFromGpu(Buffer <T> **buffer) {
	Buffer <T> *bufferRef = *buffer;
	int bufferTotalSize = bufferSize(bufferRef);
	checkCudaErrors(cudaMemcpy(bufferRef->data, bufferRef->gData, bufferTotalSize, cudaMemcpyDeviceToHost));
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
			bufferRef->data = (T *) malloc(sizeof(T) * length);
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

		void freeBuffer(Buffer<T> **buffer) {
			Buffer<T> *bufferRef = *buffer;
			if(bufferRef->data != NULL)
				free(bufferRef->data);
#ifdef __CUDACC__
            checkCudaErrors(cudaFree(bufferRef->gData));
#endif
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
			Buffer<T> *ret = (Buffer<T> *) malloc(sizeof(Buffer<T>));
            T *buffData = (T *) malloc(sizeof(T) * length);
            for(int i = 0; i < length; i++)
                buffData[i] = data[i];
            ret->data = buffData;
			ret->length = length;

#ifdef __CUDACC__
			T *gData;
	T **gDataRef = &(gData);
	checkCudaErrors(cudaMalloc((void **) gDataRef, sizeof(T) * length));
	ret->gData = gData;
	checkCudaErrors(cudaMemcpy(ret->gData, ret->data, sizeof(T) * length, cudaMemcpyHostToDevice));
#endif
			return ret;
		}
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
