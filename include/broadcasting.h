/*
 * broadcasting.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef BROADCASTING_H_
#define BROADCASTING_H_
#include <dll.h>
#include <sharedmem.h>
#include <shape.h>
#include <op.h>
#include <templatemath.h>
#include <helper_cuda.h>
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#ifdef __JNI__
#include <jni.h>
#endif
namespace functions {
    namespace broadcast {

/**
 * Broadcast operation
 * for broadcasting a smaller tensor
 * along long a bigger one.
 */
        template<typename T>
        class Broadcast: public functions::ops::Op<T> {
        public:

            /**
             *
             * @param d1
             * @param d2
             * @return
             */
            virtual
#ifdef __CUDACC__
            inline __device__  __host__

#elif defined(__GNUC__)


#endif
            T op(T d1, T d2) = 0;
            /**
             *
             * @param d1
             * @return
             */
            virtual
#ifdef __CUDACC__
            inline __device__  __host__

#elif defined(__GNUC__)


#endif
            T op(T d1) = 0;

#ifdef __CUDACC__
            __inline__ __device__ void transform(
			T *x,
			int *xShapeInfo,
			T *y,
			int *yShapeInfo,
			T *result,
			int *resultShapeInfo,
			int *dimension,
			int dimensionLength) {

		int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
		int yElementWiseStride = shape::elementWiseStride(yShapeInfo);

		//length for the tad
		int yLength = shape::length(yShapeInfo);
		//length for the tad
		int xLength = shape::length(xShapeInfo);

		int resultLength = shape::length(resultShapeInfo);
#pragma unroll
		for (int i = blockIdx.x * blockDim.x + threadIdx.x;
				i < resultLength;
				i += blockDim.x * gridDim.x) {
			int yOffset2 = ((i / xElementWiseStride) % yLength) * yElementWiseStride;
			result[i] = op(x[i],y[yOffset2]);

		}

	}
#endif

            /**
             * CPU execution
             * @param x the input
             * @param xShapeInfo the x shape information
             * @param y the y data
             * @param yShapeInfo the y shape information
             * @param result the result
             * @param resultShapeInfo the result shape information
             * @param dimension the dimension to broadcast along long
             * @param dimensionLength the length of the dimension buffer
             */
            virtual void exec(T *x, int *xShapeInfo, T *y, int *yShapeInfo, T *result,
                              int *resultShapeInfo, int *dimension, int dimensionLength) {

                int xElementWiseStride = shape::tadElementWiseStride(xShapeInfo,dimension,dimensionLength);
                int yElementWiseStride = shape::elementWiseStride(yShapeInfo);
                //length for the tad
                int yLength = shape::length(yShapeInfo);
                //length for the tad
                int xLength = shape::length(xShapeInfo);
               char xOrder = shape::order(xShapeInfo);
                if(xOrder == 'c') {
                    int *xStride = shape::stride(xShapeInfo);
                    int *xShape = shape::shapeOf(xShapeInfo);

                    //optimized loop for vectorization
                    if (xElementWiseStride == 1 && yElementWiseStride == 1) {
                        if(dimension[0] % 2 == 0) {

#pragma omp parallel for
                            for (int i = 0; i < xLength; i++) {
                                int yOffset2 =  (i  / xStride[0]) * yElementWiseStride;
                                result[i] = op(x[i], y[yOffset2]);


                            }
                        }
                        else {

#pragma omp parallel for
                            for (int i = 0; i < xLength; i++) {
                                int yOffset2 =  (i % xShape[dimension[0]]) * yElementWiseStride;
                                result[i] = op(x[i], y[yOffset2]);


                            }
                        }

                    }

                    else {
                        if(dimension[0] % 2 == 0) {
#pragma omp parallel for
                            for (int i = 0; i < xLength; i++) {
                                int yOffset2 =  (i  / xStride[dimension[0]]) * yElementWiseStride;
                                result[i] = op(x[i], y[yOffset2]);

                            }
                        }
                        else {
#pragma omp parallel for
                            for (int i = 0; i < xLength; i++) {
                                int yOffset2 =  (i % xShape[dimension[0]]) * yElementWiseStride;
                                result[i] = op(x[i], y[yOffset2]);

                            }
                        }


                    }

                }

                else if(xOrder == 'f') {
                    if (xElementWiseStride == 1 && yElementWiseStride == 1) {
#pragma omp parallel for
                        for (int i = 0; i < xLength; i++) {
                            int yOffset2 =  (i % yLength) * yElementWiseStride;
                            result[i] = op(x[i], y[yOffset2]);


                        }
                    }

                    else {
#pragma omp parallel for
                        for (int i = 0; i < xLength; i++) {
                            int yOffset2 =  (i % yLength) * yElementWiseStride;
                            result[i] = op(x[i], y[yOffset2]);
                        }
                    }
                }

            }

            virtual inline
#ifdef __CUDACC__
            __host__ __device__
#endif
            void aggregateExtraParams(T **extraParamsTotal,T **extraParamsLocal) {
                //no extra params aggregation needs to happen
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)

#endif
            virtual ~Broadcast() {
            }
#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)

#endif
            Broadcast() {
            }

        };

        namespace ops {
            template<typename T>
            class Add: public  functions::broadcast::Broadcast<T> {
            public:


                /**
                 *
                 * @param d1
                 * @param d2
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T d2) {
                    return d1 + d2;
                }
                /**
                 *
                 * @param d1
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                virtual ~Add() {
                }
            };

            template<typename T>
            class Copy: public virtual functions::broadcast::Broadcast<T> {
            public:


                /**
                 *
                 * @param d1
                 * @param d2
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T d2) {
                    return d2;
                }
                /**
                 *
                 * @param d1
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                virtual ~Copy() {
                }

            };

            template<typename T>
            class Divide: public virtual functions::broadcast::Broadcast<T> {
            public:


                /**
                 *
                 * @param d1
                 * @param d2
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T d2) {
                    return d1 / d2;
                }
                /**
                 *
                 * @param d1
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                virtual ~Divide() {
                }

            };

            template<typename T>
            class Multiply: public virtual functions::broadcast::Broadcast<T> {
            public:


                /**
                 *
                 * @param d1
                 * @param d2
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T d2) {
                    return d1 * d2;
                }
                /**
                 *
                 * @param d1
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                virtual ~Multiply() {
                }

            };

            template<typename T>
            class ReverseDivide: public virtual functions::broadcast::Broadcast<T> {
            public:


                /**
                 *
                 * @param d1
                 * @param d2
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T d2) {
                    return d2 / d1;
                }
                /**
                 *
                 * @param d1
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                virtual ~ReverseDivide() {
                }

            };

            template<typename T>
            class ReverseSubtract: public virtual functions::broadcast::Broadcast<T> {
            public:


                /**
                 *
                 * @param d1
                 * @param d2
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T d2) {
                    return d2 - d1;
                }
                /**
                 *
                 * @param d1
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                virtual ~ReverseSubtract() {
                }

            };

            template<typename T>
            class Subtract: public virtual functions::broadcast::Broadcast<T> {
            public:

                /**
                 *
                 * @param d1
                 * @param d2
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1, T d2) {
                    return d1 - d2;
                }
                /**
                 *
                 * @param d1
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)


#endif
                T op(T d1) {
                    return d1;
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

#endif
                virtual ~Subtract() {
                }

            };
        }

        template<typename T>
        class BroadcastOpFactory {
        public:

#ifdef __CUDACC__
            __host__ __device__
#endif
            BroadcastOpFactory() {
            }


            /**
             * creates an operation
             * @param op the op number to create:
             * 0: Add
             * 1: Subtract
             * 2: Multiply
             * 3: Divide
             * 4: ReverseDivide
             * 5: Reverse Subtract
             * 6: Copy
             * @return the broadcast operation
             */
#ifdef __CUDACC__
            __inline__ __host__ __device__
#endif
            Broadcast<T> * getOp(int op) {
                if (op == 0) {
                    return new functions::broadcast::ops::Add<T>();
                } else if (op == 1) {
                    return new functions::broadcast::ops::Subtract<T>();
                } else if (op == 2) {
                    return new  functions::broadcast::ops::Multiply<T>();
                } else if (op == 3) {
                    return new functions::broadcast::ops::Divide<T>();
                } else if (op == 4) {
                    return new functions::broadcast::ops::ReverseDivide<T>();
                } else if (op == 5) {
                    return new functions::broadcast::ops::ReverseSubtract<T>();
                } else if (op == 6) {
                    return new functions::broadcast::ops::Copy<T>();
                }

                return NULL;

            }

        };

    }
}

#ifdef __CUDACC__

/**
 * Meant to be called from an external interface
 * and the driver api
 * @param opNum the op number to execute
 * @param x the input data
 * @param xShapeInfo the x shape info for input
 * @param y the y to broadcast
 * @param yShapeInfo the shape information of the broadcast info
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result buffer
 * @param dimension the dimension(s) to do broadcast along long
 * @param dimensionLength the length of the dimension buffer
 * @param gpuInformation the gpu information such as blockdim,griddim and shared
 * memory size
 */
template <typename T>
__device__ void broadcastGeneric(
		int opNum,
		T *x,
		int *xShapeInfo,
		T *y,
		int *yShapeInfo,
		T *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength) {

	//TODO: Reduce object creation
	__shared__ functions::broadcast::Broadcast<T> *op;
	__shared__ functions::broadcast::BroadcastOpFactory<T> *newOpFactory;
	if(threadIdx.x == 0)
		newOpFactory =  new functions::broadcast::BroadcastOpFactory<T>();
	__syncthreads();
	if(threadIdx.x == 0) {
		op = newOpFactory->getOp(opNum);
	}
	__syncthreads();


	op->transform(
			x,
			xShapeInfo,
			y,
			yShapeInfo,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength);

	if(threadIdx.x == 0) {
		free(op);
		free(newOpFactory);
	}
}

/**
 * Meant to be called from an external interface
 * and the driver api
 * @param opNum the op number to execute
 * @param x the input data
 * @param xShapeInfo the x shape info for input
 * @param y the y to broadcast
 * @param yShapeInfo the shape information of the broadcast info
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result buffer
 * @param dimension the dimension(s) to do broadcast along long
 * @param dimensionLength the length of the dimension buffer
 * @param gpuInformation the gpu information such as blockdim,griddim and shared
 * memory size
 */
extern "C" __global__ void broadcastDouble(
		int opNum,
		double *x, int *xShapeInfo,
		double *y, int *yShapeInfo,
		double *result, int *resultShapeInfo,
		int *dimension,
		int dimensionLength) {
	broadcastGeneric<double>(
			opNum,
			x,
			xShapeInfo,
			y,
			yShapeInfo,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength);

}


/**
 * Meant to be called from an external interface
 * and the driver api
 * @param opNum the op number to execute
 * @param x the input data
 * @param xShapeInfo the x shape info for input
 * @param y the y to broadcast
 * @param yShapeInfo the shape information of the broadcast info
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result buffer
 * @param dimension the dimension(s) to do broadcast along long
 * @param dimensionLength the length of the dimension buffer
 * @param gpuInformation the gpu information such as blockdim,griddim and shared
 * memory size
 */
extern "C" __global__ void broadcastFloat(
		int opNum,
		float *x, int *xShapeInfo,
		float *y, int *yShapeInfo,
		float *result, int *resultShapeInfo,
		int *dimension,
		int dimensionLength) {
	broadcastGeneric<float>(
			opNum,
			x,
			xShapeInfo,
			y,
			yShapeInfo,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength);

}

#endif



#endif /* BROADCASTING_H_ */
