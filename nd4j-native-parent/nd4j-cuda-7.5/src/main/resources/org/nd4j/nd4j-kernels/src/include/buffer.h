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

namespace nd4j {
    namespace buffer {
        template<typename T>
        struct Buffer {
            int length;
            T *data;
            T *gData;
        };

        template<>
        struct Buffer<double> {
            int length;
            double *data;
            double *gData;
        };


        template<>
        struct Buffer<float> {
            int length;
            float *data;
            float *gData;
        };

        template<>
        struct Buffer<int> {
            int length;
            int *data;
            int *gData;
        };

/**
 * Returns the size of the buffer
 * in bytes
 * @param buffer the buffer to get the size of
 * @return the size of the buffer in bytes
 */
        template<typename T>
        __device__ __host__

        size_t bufferSize(Buffer<T> *buffer);

/**
 * Copies data to the gpu
 * @param buffer the buffer to copy
 */
        template<typename T>
        __host__ void copyDataToGpu(Buffer<T> **buffer);

/**
 * Copies data from the gpu
 * @param buffer the buffer to copy
 */
        template<typename T>
        __host__ void copyDataFromGpu(Buffer<T> **buffer);

/**
 * Allocate buffer of the given
 * length on the cpu and gpu.
 */
        template<typename T>
        __host__ void allocBuffer(Buffer<T> **buffer, int length);


/**
 * Frees the given buffer
 * (gpu and cpu
 */
        template<typename T>
        __host__ void freeBuffer(Buffer<T> **buffer);

/**
 * Creates a buffer
 * based on the data
 * and also synchronizes
 * the data on the gpu.
 */
        template<typename T>
        __host__ Buffer<T>
        *
        createBuffer(T
        *data,
        int length
        );

/**
 * Print the buffer on the host
 * @param buff
 */
        template<typename T>
        __host__ void printArr(Buffer<T> *buff);

    }
}


#endif /* BUFFER_H_ */
