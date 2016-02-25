/*
 * array.h
 *
 *  Created on: Dec 24, 2015
 *      Author: agibsonccc
 */



#ifndef ARRAY_H_
#define ARRAY_H_

#include <shape.h>

namespace nd4j {
    namespace array {

        template<typename T>
        struct NDArray {
            T *data;
            T *gData;
            int *shape, *stride;
            int *gShape, *gStride;
            int offset;
            int rank;
            char ordering;
        };

        template<>
        struct NDArray<double> {
            double *data;
            double *gData;
            int *shape, *stride;
            int *gShape, *gStride;
            int offset;
            int rank;
            char ordering;
        };


        template<>
        struct NDArray<float> {
            float *data;
            float *gData;
            int *shape, *stride;
            int *gShape, *gStride;
            int offset;
            int rank;
            char ordering;
        };

/**
 * Returns the length of this ndarray
 */
        template<typename T>
        __device__ __host__

        size_t length(NDArray<T> *arr);

        /**
         * Returns the length of
         * this ndarray
         * in bytes
         */
        template<typename T>
        __device__ __host__

        size_t lengthInBytes(NDArray<T> *arr);


        /**
         * Creates an ndarray
         * from the given rank,shape,stride,
         * offset and fills the array
         * with the given default value
         */
        template<typename T>
        __device__ __host__

        NDArray<T> *createFrom(int rank, int *shape, int *stride, int offset, T defaultValue);


        /**
         * Copy the already allocated host pointers
         * to the gpu.
         *
         * Note that the ndarray must
         * have already been initialized.
         *
         */
        template<typename T>
        __host__ void allocateNDArrayOnGpu(NDArray<T> **arr);


        /**
         * Creates an ndarray based on the
         * given parameters
         * and then allocates it on the gpu
         */
        template<typename T>
        __host__ NDArray<T>
        *

        createFromAndAllocateOnGpu(int rank, int *shape, int *stride, int offset, T defaultValue);


        /**
         * Copies the host data
         * from the gpu
         * to the cpu
         * for the given ndarray
         */
        template<typename T>
        __host__ void copyFromGpu(NDArray<T> **arr);


        template<typename T>
        __host__ void freeNDArrayOnGpuAndCpu(NDArray<T> **arr);


        /**
         * Allocate the data based
         * on the shape information
         */
        template<typename T>
        __host__ __device__

        void allocArrayData(NDArray<T> **arr);


        /**
         * Returns the shape information for this array
         * Note that this allocates memory that should be freed.
         *
         * Note that it will use the pointers directly by reference
         * for shape and stride
         * @return the shape information for the given array
         */
        template<typename T>
        __host__ __device__

        shape::ShapeInformation *shapeInfoForArray(NDArray<T> *arr);


        /**
         * Create based on the given shape information
         * and specified default value.
         * Note that the shape information is assumed to be filled in.
         *
         *
         */
        template<typename T>
        __host__ __device__

        NDArray<T> *createFromShapeInfo(shape::ShapeInformation *info, T defaultValue);

        /**
         * Print the array on the gpu
         * @param arr
         */
        template<typename T>
        __device__ void printArrGpu(NDArray<T> *arr);

        /**
         * Print the array on the host
         * @param arr
         */
        template<typename T>
        __host__ void printArrHost(NDArray<T> *arr);


    }
}


#endif /* ARRAY_H_ */
