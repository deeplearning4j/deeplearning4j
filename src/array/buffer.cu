/*
 * buffer_impl.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#include <buffer.h>

namespace nd4j {
    namespace buffer {
/**
 *
 * @param buffer
 * @return
 */
        template<typename T>
#ifdef __CUDACC__
        __host__ __device__
#endif

        size_t bufferSize(Buffer <T> *buffer) {
            return sizeof(T) * buffer->length;
        }


/**
 *
 * @param buffer
 */
        template<typename T>
#ifdef __CUDACC__
        __host__
#endif
        void copyDataToGpu(Buffer <T> **buffer) {
            Buffer <T> *bufferRef = *buffer;
#ifdef __CUDACC__
            checkCudaErrors(
                    cudaMemcpy(bufferRef->gData, bufferRef->data, bufferSize(bufferRef), cudaMemcpyHostToDevice));
#endif
        }

/**
 *
 * @param buffer
 */
        template<typename T>
#ifdef __CUDACC__
        __host__
#endif
        void copyDataFromGpu(Buffer <T> **buffer) {
            Buffer <T> *bufferRef = *buffer;
            int bufferTotalSize = bufferSize(bufferRef);
#ifdef __CUDACC__
            checkCudaErrors(cudaMemcpy(bufferRef->data, bufferRef->gData, bufferTotalSize, cudaMemcpyDeviceToHost));
#endif
        }

/**
 * Allocate buffer of the given
 * length on the cpu and gpu.
 */
        template<typename T>
#ifdef __CUDACC__
        __host__
#endif
        void allocBuffer(Buffer <T> **buffer, int length) {
            Buffer <T> *bufferRef = *buffer;
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
        void freeBuffer(Buffer <T> **buffer) {
            Buffer <T> *bufferRef = *buffer;
            delete[] bufferRef->data;
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
        Buffer<T>
        *
        createBuffer(T
                     *data,
                     int length
        ) {
            Buffer<T> *ret = (Buffer<T> * )
                    malloc(sizeof(Buffer<T>));
            ret->
                    data = data;
            ret->
                    length = length;
            T *gData;
            T **gDataRef = &(gData);
#ifdef __CUDACC__
            checkCudaErrors(cudaMalloc((void **) gDataRef, sizeof(T) * length));
            ret->
                    gData = gData;
            checkCudaErrors(cudaMemcpy(ret->gData, ret->data, sizeof(T) * length, cudaMemcpyHostToDevice));
#endif
            return ret;
        }


        template<typename T>
#ifdef __CUDACC__
        __host__
#endif
        void printArr(Buffer <T> *buff) {
            for (int i = 0; i < buff->length; i++) {
                printf("Buffer[%d] was %f\n", i, buff->data[i]);
            }
        }

    }
}


