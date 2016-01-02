/*
 * array_impl.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#include <array.h>
#include <helper_string.h>
#include <helper_cuda.h>

namespace nd4j {
    namespace array {
/**
 * Returns the length of this ndarray
 */
        template<typename T>
#ifdef __CUDACC__
        __host__ __device__
#endif

        size_t length(NDArray<T> *arr) {
            size_t size = shape::prod(arr->shape, arr->rank);
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

        size_t lengthInBytes(NDArray<T> *arr) {
            size_t size = shape::prod(arr->shape, arr->rank) * sizeof(T);
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
        __host__ __device__
#endif

        NDArray<T> *createFrom(int rank, int *shape, int *stride, int offset, T defaultValue) {
            NDArray<T> *ret = (NDArray<T> *) malloc(sizeof(NDArray<T>));
            ret->rank = rank;
            ret->shape = shape;
            ret->stride = stride;
            ret->offset = offset;
            size_t size = lengthInBytes(ret);
            ret->data = (T *) malloc(size);
            memset(ret->data, defaultValue, size);
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
        template<typename T>
#ifdef __CUDACC__
        __host__
#endif
        void allocateNDArrayOnGpu(NDArray<T> **arr) {
            NDArray<T> *arrRef = *arr;
            T *gData = 0;
            size_t size = lengthInBytes(arrRef);
#ifdef __CUDACC__
            checkCudaErrors(cudaMalloc(&gData, size));
            checkCudaErrors(cudaMemcpy(gData, arrRef->data, size, cudaMemcpyHostToDevice));
            arrRef->gData = gData;
#endif
            size_t intRankSize = arrRef->rank * sizeof(int);
#ifdef __CUDACC__
            int *gShape = 0, *gStride = 0;
            checkCudaErrors(cudaMalloc(&gShape, intRankSize));
            checkCudaErrors(cudaMemcpy(gShape, arrRef->shape, intRankSize, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMalloc(&gStride, intRankSize));
            checkCudaErrors(cudaMemcpy(gStride, arrRef->stride, intRankSize, cudaMemcpyHostToDevice));
            arrRef->gShape = gShape;
            arrRef->gStride = gStride;
#endif
        }

/**
 * Creates an ndarray based on the
 * given parameters
 * and then allocates it on the gpu
 */
        template<typename T>
#ifdef __CUDACC__
        __host__
#endif
        NDArray<T>
        *

        createFromAndAllocateOnGpu(int rank, int *shape, int *stride, int offset, T defaultValue) {
            NDArray<T> *ret = createFrom(rank, shape, stride, offset, defaultValue);
            allocateNDArrayOnGpu(&ret);
            return ret;
        }


/**
 * Copies the host data
 * from the gpu
 * to the cpu
 * for the given ndarray
 */
        template<typename T>
#ifdef __CUDACC__
        __host__
#endif
        void copyFromGpu(NDArray<T> **arr) {
            NDArray<T> *arrRef = *arr;
#ifdef __CUDACC__
            checkCudaErrors(cudaMemcpy(arrRef->data, arrRef->gData, lengthInBytes(arrRef), cudaMemcpyDeviceToHost));
#endif
        }


        template<typename T>
#ifdef __CUDACC__
        __host__
#endif
        void freeNDArrayOnGpuAndCpu(NDArray<T> **arr) {
            NDArray<T> *arrRef = *arr;
            delete[] arrRef->data;
#ifdef __CUDACC__
            checkCudaErrors(cudaFree(arrRef->gData));
#endif
            delete[] arrRef->shape;
#ifdef __CUDACC__
            checkCudaErrors(cudaFree(arrRef->gShape));
#endif
            delete[] arrRef->stride;
#ifdef __CUDACC__
            checkCudaErrors(cudaFree(arrRef->gStride));
#endif
        }


/**
 * Allocate the data based
 * on the shape information
 */
        template<typename T>
#ifdef __CUDACC__
        __host__ __device__
#endif

        void allocArrayData(NDArray<T> **arr) {
            NDArray<T> *arrRef = *arr;
            int dataLength = shape::prod(arrRef->shape, arrRef->rank);
            arrRef->data = (buffer::Buffer<T> *) malloc(sizeof(T) * dataLength);
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

        shape::ShapeInformation *shapeInfoForArray(NDArray<T> *arr) {
            shape::ShapeInformation *info = (shape::ShapeInformation *) malloc(sizeof(shape::ShapeInformation));
            info->offset = arr->offset;
            info->order = arr->ordering;
            info->shape = arr->shape;
            info->stride = arr->stride;
            info->rank = arr->rank;
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
        __host__ __device__
#endif

        NDArray<T> *createFromShapeInfo(shape::ShapeInformation *info, T defaultValue) {
            return createFrom(info->rank, info->shape, info->stride, info->offset, defaultValue);
        }


        template<typename T>
#ifdef __CUDACC__
        __device__
#endif
        void printArrGpu(NDArray<T> *arr) {
#ifdef __CUDACC__
            for (int i = 0; i < length(arr); i++) {
                printf("Arr[%d] is %f\n", arr->gData[i]);
            }
        }
#endif

            template<typename T>
#ifdef __CUDACC__
            __host__
#endif
            void printArrHost(NDArray<T> *arr) {
                for (int i = 0; i < length(arr); i++) {
                    printf("Arr[%d] is %f\n", i, arr->data[i]);
                }
            }


        }
    }
}


