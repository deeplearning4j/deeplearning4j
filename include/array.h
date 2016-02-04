/*
 * array.h
 *
 *  Created on: Dec 24, 2015
 *      Author: agibsonccc
 */

#ifndef ARRAY_H_
#define ARRAY_H_

#include <shape.h>
#include <helper_string.h>
#include <helper_cuda.h>
#include <math.h>
#include <buffer.h>

namespace nd4j {
    namespace array {

#define EPSILON 1e-6

        template<typename T>
        struct NDArray {
            buffer::Buffer<T> *data;
            buffer::Buffer<int> *shape;
            buffer::Buffer<int> *stride;
            int offset;
            int rank;
            char ordering;

            int operator==(const NDArray<T>& other) {
                if (rank != other.rank)
                    return 0;

                int length = shape::prod(shape->data, rank);
                int otherLength = shape::prod(other.shape->data, other.rank);
                if (length != otherLength)
                    return 0;
                if (ordering != other.ordering)
                    return 0;
                if (offset != other.offset)
                    return 0;
                for (int i = 0; i < rank; i++) {
                    if (shape->data[i] != other.shape->data[i] || stride->data[i] != other.stride->data[i])
                        return 0;
                }
                for (int i = 0; i < length; i++) {
                    T diff = (T) data[i] - other.data[i];
                    if (abs(diff) >= EPSILON)
                        return 0;
                }

                return 1;
            }
        };

        template<typename T>
        class NDArrays {
        public:
            /**
             * Creates an ndarray
             * from the given rank,shape,stride,
             * offset and fills the array
             * with the given default value
             */

#ifdef __CUDACC__
            __host__
#endif

            static NDArray<T> * createFrom(T *data,int rank, int *shape, int *stride,	int offset);
            /**
             * Returns the length of this ndarray
             */
            static
#ifdef __CUDACC__
            __device__  __host__

#endif
            size_t length(NDArray<T> *arr);

            /**
             * Returns the length of
             * this ndarray
             * in bytes
             */
            static
#ifdef __CUDACC__
            __device__  __host__

#endif
            size_t lengthInBytes(NDArray<T> *arr);

            /**
             * Creates an ndarray
             * from the given rank,shape,stride,
             * offset and fills the array
             * with the given default value
             */
            static
#ifdef __CUDACC__
            __host__

#endif
            NDArray<T> *createFrom(int rank, int *shape, int *stride, int offset,
                                   T defaultValue);

            /**
             * Copy the already allocated host pointers
             * to the gpu.
             *
             * Note that the ndarray must
             * have already been initialized.
             *
             */
            static
#ifdef __CUDACC__
            __host__
#endif
            void allocateNDArrayOnGpu(NDArray<T> **arr);

            /**
             * Creates an ndarray based on the
             * given parameters
             * and then allocates it on the gpu
             */

#ifdef __CUDACC__
            static __host__ NDArray<T> * createFromAndAllocateOnGpu(int rank, int *shape, int *stride, int offset, T defaultValue);

#endif
            /**
             * Copies the host data
             * from the gpu
             * to the cpu
             * for the given ndarray
             */
            static
#ifdef __CUDACC__
            __host__
#endif
            void copyFromGpu(NDArray<T> **arr);

            /**
             * Frees data on gpu and cpu
             * @param arr
             */
            static
#ifdef __CUDACC__
            __host__
#endif
            void freeNDArrayOnGpuAndCpu(NDArray<T> **arr);

            /**
             * Allocate the data based
             * on the shape information
             */
#ifdef __CUDACC__
            __host__ __device__
#endif

            static void allocArrayData(NDArray<T> **arr);

            /**
             * Returns the shape information for this array
             * Note that this allocates memory that should be freed.
             *
             * Note that it will use the pointers directly by reference
             * for shape and stride
             * @return the shape information for the given array
             */
#ifdef __CUDACC__
            __host__ __device__
#endif

            static shape::ShapeInformation *shapeInfoForArray(NDArray<T> *arr);

            /**
             * Create based on the given shape information
             * and specified default value.
             * Note that the shape information is assumed to be filled in.
             *
             *
             */
#ifdef __CUDACC__
            __host__
#endif

            static NDArray<T> *createFromShapeInfo(shape::ShapeInformation *info,
                                                   T defaultValue);

            /**
             * Print the array on the gpu
             * @param arr
             */
            static
#ifdef __CUDACC__
            __device__
#endif
            void printArrGpu(NDArray<T> *arr);

            /**
             * Print the array on the host
             * @param arr
             */
            static
#ifdef __CUDACC__
            __host__
#endif
            void printArrHost(NDArray<T> *arr);

        };

/**
 * Returns the length of this ndarray
 */
        template<typename T>
#ifdef __CUDACC__
        __host__ __device__
#endif

        size_t NDArrays<T>::length(NDArray<T> *arr) {
            size_t size = shape::prod(arr->shape->data, arr->rank);
            return size;
        }

/**
 * Returns the length of
 * this ndarray
 * in bytes
 */
        template<typename T>
#ifdef __CUDACC__
        __host__ __device__
#endif

        size_t NDArrays<T>::lengthInBytes(NDArray<T> *arr) {
            size_t size = shape::prod(arr->shape->data, arr->rank) * sizeof(T);
            return size;
        }



/**
 * Creates an ndarray
 * from the given rank,shape,stride,
 * offset and fills the array
 * with the given default value
 */
        template<typename T>

#ifdef __CUDACC__
        __host__
#endif

        NDArray<T> * NDArrays<T>::createFrom(
                T *data,
                int rank,
                int *shape,
                int *stride,int offset) {
            NDArray<T> *ret = (NDArray<T> *) malloc(sizeof(NDArray<T>));
            ret->rank = rank;

            ret->shape = nd4j::buffer::createBuffer(shape,rank);
            ret->stride = nd4j::buffer::createBuffer(stride,rank);
            ret->offset = offset;
            size_t size = lengthInBytes(ret);
            int length = size / sizeof(T);
            ret->data = nd4j::buffer::createBuffer<T>(data, length);
            return ret;
        }

/**
 * Creates an ndarray
 * from the given rank,shape,stride,
 * offset and fills the array
 * with the given default value
 */
        template<typename T>

#ifdef __CUDACC__
        __host__
#endif

        NDArray<T> * NDArrays<T>::createFrom(int rank, int *shape, int *stride,
                                             int offset, T defaultValue) {
            printf("Mallocing array\n");
            NDArray<T> *ret = (NDArray<T> *) malloc(sizeof(NDArray<T>) * 2);
            printf("Malloced array\n");
            ret->rank = rank;
            printf("About to create shape buffer of rank %d\n",rank);
            ret->shape = nd4j::buffer::createBuffer(shape,rank);
            printf("Created shape buffer of rank %d\n",rank);
            printf("About to create stride buffer of rank %d\n",rank);
            ret->stride = nd4j::buffer::createBuffer(stride,rank);
            printf("Created stride buffer of rank %d\n",rank);
            ret->offset = offset;
            printf("About to compute size of bytes\n");
            size_t size = lengthInBytes(ret);
            printf("Computed size in bytes\n");
            int length = size / sizeof(T);
            printf("Mallocing data of size %d\n",size);
            int realSize = shape::prod(shape,rank);
            printf("length of array is %d\n",realSize);
            T * data = (T *) malloc(realSize * sizeof(T));
            printf("Malloced data for array\n");
            ret->data = nd4j::buffer::createBuffer<T>(data, length);
            printf("Assigning data\n");
            ret->data->assign(data);
            return ret;
        }

/**
 * Copy the already allocated host pointers
 * to the gpu.
 *
 * Note that the ndarray must
 * have already been initialized.
 *
 */
#ifdef __CUDACC__
        template<typename T>

__host__

void NDArrays<T>::allocateNDArrayOnGpu(NDArray <T> **arr) {
	NDArray <T> *arrRef = *arr;
	size_t size = lengthInBytes(arrRef);
	nd4j::buffer::copyDataToGpu(&((*arr)->data));
	nd4j::buffer::copyDataToGpu(&((*arr)->shape));
	nd4j::buffer::copyDataToGpu(&((*arr)->stride));
}
#endif
/**
 * Creates an ndarray based on the
 * given parameters
 * and then allocates it on the gpu
 */
#ifdef __CUDACC__
        template<typename T>
__host__

NDArray<T> * NDArrays<T>::createFromAndAllocateOnGpu(int rank, int *shape, int *stride, int offset, T defaultValue) {
	NDArray<T> * ret = createFrom(rank, shape, stride, offset, defaultValue);
	allocateNDArrayOnGpu(&ret);
	return ret;
}
#endif

/**
 * Copies the host data
 * from the gpu
 * to the cpu
 * for the given ndarray
 */
#ifdef __CUDACC__
        template<typename T>

__host__

void NDArrays<T>::copyFromGpu(NDArray<T> **arr) {
	NDArray<T> * arrRef = *arr;
	checkCudaErrors(cudaMemcpy(arrRef->data->data, arrRef->data->gData, lengthInBytes(arrRef), cudaMemcpyDeviceToHost));
}

#endif
        template<typename T>
#ifdef __CUDACC__
        __host__
#endif
        void NDArrays<T>::freeNDArrayOnGpuAndCpu(NDArray<T> **arr) {
            NDArray<T> * arrRef = *arr;
            nd4j::buffer::Buffer<T> *dataBuf = arrRef->data;
            nd4j::buffer::freeBuffer<T>(&dataBuf);
            nd4j::buffer::freeBuffer(&arrRef->shape);
            nd4j::buffer::freeBuffer(&arrRef->stride);

        }

/**
 * Allocate the data based
 * on the shape information
 */
        template<typename T>
#ifdef __CUDACC__
        __host__ __device__
#endif

        void NDArrays<T>::allocArrayData(NDArray<T> * *arr) {
            NDArray<T> * arrRef = *arr;
            int dataLength = shape::prod(arrRef->shape->data,arrRef->rank);
            arrRef->data = (T *) malloc(sizeof(T) * dataLength);
        }

/**
 * Returns the shape information for this array
 * Note that this allocates memory that should be freed.
 *
 * Note that it will use the pointers directly by reference
 * for shape and stride
 * @return the shape information for the given array
 */
        template<typename T>
#ifdef __CUDACC__
        __host__ __device__
#endif

        shape::ShapeInformation * NDArrays<T>::shapeInfoForArray(NDArray<T> * arr) {
            shape::ShapeInformation *info = (shape::ShapeInformation *) malloc(
                    sizeof(shape::ShapeInformation));
            info->offset = arr->offset;
            info->order = arr->ordering;
            info->shape = arr->shape->data;
            info->stride = arr->stride->data;
            info->rank = arr->rank;
            int elementWiseStride = shape::computeElementWiseStride(arr->rank,
                                                                    arr->shape->data, arr->stride->data, arr->ordering == 'c');
            info->elementWiseStride = elementWiseStride;
            return info;
        }

/**
 * Create based on the given shape information
 * and specified default value.
 * Note that the shape information is assumed to be filled in.
 *
 *
 */
        template<typename T>
#ifdef __CUDACC__
        __host__
#endif

        NDArray<T> * NDArrays<T>::createFromShapeInfo(shape::ShapeInformation *info,
                                                      T defaultValue) {
            return createFrom(info->rank, info->shape, info->stride, info->offset,
                              defaultValue);
        }

#ifdef __CUDACC__
        template<typename T>

__device__

void NDArrays<T>::printArrGpu(NDArray<T> *arr) {
	for (int i = 0; i < length(arr); i++) {
		printf("Arr[%d] is %f\n", arr->gData[i]);
	}
}
#endif

        template<typename T>
#ifdef __CUDACC__
        __host__
#endif
        void NDArrays<T>::printArrHost(NDArray<T> *arr) {
            for (int i = 0; i < length(arr); i++) {
                printf("Arr[%d] is %f\n", i, arr->data[i]);
            }
        }

    }
}

#endif /* ARRAY_H_ */
