/*
 * transform.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef TRANSFORM_H_
#define TRANSFORM_H_
#include <templatemath.h>
#include <op.h>
#include <omp.h>
#ifdef __CUDACC__
#include <helper_cuda.h>
#endif

#ifdef __JNI__
#include <jni.h>
#endif
namespace functions {
    namespace transform {

        template<typename T>
        class Transform : public functions::ops::Op<T> {
        public:

            /**
             * The op for transforms
             * @param d1
             * @param params
             * @return
             */
            virtual
#ifdef __CUDACC__
            inline __device__ __host__

#elif defined(__GNUC__)
            __always_inline

#endif
            T op(T d1, T *params) = 0;

#ifdef __CUDACC__
            /**
	 * Cuda implementation of transform
	 * @param dx
	 * @param xShapeInfo
	 * @param result
	 * @param resultShapeInfo
	 * @param extraParams
	 * @param n
	 */
	virtual __inline__ __device__ void transform(
			T *dy,
			int *shapeInfo,
			T *params,
			T *result,
			int *indexes) {

		int n = shape::length(shapeInfo);
		int totalThreads = gridDim.x * blockDim.x;
		int tid = threadIdx.x;
		int i = blockIdx.x * blockDim.x + tid;

		/* equal, positive, non-unit increments. */
#pragma unroll
		for (; i < n; i+= totalThreads) {
			result[indexes[i]] = op(dy[indexes[i]], params);
		}

	}


	/**
	 * Cuda implementation of transform
	 * @param dx
	 * @param xShapeInfo
	 * @param result
	 * @param resultShapeInfo
	 * @param extraParams
	 * @param n
	 */
	virtual __inline__ __device__ void transform(
			T *dy,
			int *shapeInfo,
			T *params,
			T *result) {
		int *xShape = shape::shapeOf(shapeInfo);
		int *xStride = shape::stride(shapeInfo);
		char xOrder = shape::order(shapeInfo);
		int n = shape::length(shapeInfo);
		int xRank = shape::rank(shapeInfo);
		int xOffset = shape::offset(shapeInfo);

		int xElementWiseStride = shape::computeElementWiseStride(xRank,xShape,xStride,xOrder == 'f');

		int totalThreads = gridDim.x * blockDim.x;
		int tid = threadIdx.x;
		int i = blockIdx.x * blockDim.x + tid;
		__shared__ int length;
		if(tid == 0)
			length = shape::length(shapeInfo);
		__syncthreads();

		if(xElementWiseStride >= 1) {
			transform(length,dy,xElementWiseStride,params,result);
		}
		else {
			/* equal, positive, non-unit increments. */
#pragma unroll
			for (; i < n; i+= totalThreads) {
				int *xIdx = shape::ind2sub(xRank, xShape, i);
				int xOffset2 = shape::getOffset(xOffset, xShape, xStride, xIdx, xRank);
				result[xOffset2] = op(dy[xOffset2], params);
				free(xIdx);
			}
		}
	}

	/**
	 * Cuda implementation of transform
	 * @param dx
	 * @param xShapeInfo
	 * @param result
	 * @param resultShapeInfo
	 * @param extraParams
	 * @param n
	 */
	virtual  __inline__ __device__ void transform(int n, T *dy, int incy, T *params, T *result) {
		int totalThreads = gridDim.x * blockDim.x;
		int tid = threadIdx.x;
		int i = blockIdx.x * blockDim.x + tid;
		/* equal, positive, non-unit increments. */
#pragma unroll
		for (; i < n; i += totalThreads) {
			result[i * incy] = op(dy[i * incy], params);
		}

	}
#endif

            /**
             * CPU execution
             * @param dx the input
             * @param xStride the stride to iterate for the input
             * @param result the result buffer
             * @param resultStride the stride for result
             * storage
             * @param extraParams the extra parameters
             * @param n the number of elements to iterate on
             */
            virtual void exec(
                    T *dx,
                    int *xShapeInfo,
                    T *result,
                    int *resultShapeInfo,
                    T *extraParams,
                    const int n,
                    int *indexes) {
#pragma omp simd
                for (int i = 0; i < n; i++) {
                    result[indexes[i]] = op(dx[indexes[i]], extraParams);
                }
            }

            /**
         * CPU execution
         * @param dx the input
         * @param xStride the stride to iterate for the input
         * @param result the result buffer
         * @param resultStride the stride for result
         * storage
         * @param extraParams the extra parameters
         * @param n the number of elements to iterate on
         */
            virtual void exec(
                    T *dx,
                    int *xShapeInfo,
                    T *result,
                    int *resultShapeInfo,
                    T *extraParams,
                    const int n,
                    int *indexes,
                    int *resultIndexes) {
#pragma omp simd
                for (int i = 0; i < n; i++) {
                    result[resultIndexes[i]] = op(dx[indexes[i]], extraParams);
                }
            }


            /**
             * CPU execution
             * @param dx the input
             * @param xStride the stride to iterate for the input
             * @param result the result buffer
             * @param resultStride the stride for result
             * storage
             * @param extraParams the extra parameters
             * @param n the number of elements to iterate on
             */
            virtual void exec(
                    T *dx,
                    int *xShapeInfo,
                    T *result,
                    int *resultShapeInfo,
                    T *extraParams,
                    const int n) {


                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                int resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);
                if(xElementWiseStride >= 1 && resultElementWiseStride >= 1) {
                    exec(dx,xElementWiseStride,result,resultElementWiseStride,extraParams,n);
                }
                else {

                    int i;

                    int *xShape = shape::shapeOf(xShapeInfo);
                    int *resultShape = shape::shapeOf(resultShapeInfo);

                    int *xStride = shape::stride(xShapeInfo);
                    int *resultStride = shape::stride(resultShapeInfo);
                    int xRank = shape::rank(xShapeInfo);
                    int resultRank = shape::rank(resultShapeInfo);

                    int xOffset = shape::offset(xShapeInfo);
                    int resultOffset = shape::offset(resultShapeInfo);

                    char xOrder = shape::order(xShapeInfo);
                    char resultOrder = shape::order(xShapeInfo);
#pragma omp parallel private(i)
                    {
#pragma omp simd
                        for (i = omp_get_thread_num(); i < n; i+= omp_get_num_threads()) {
                            int *xIdx = shape::ind2sub(xRank, xShape, i);
                            int *resultIdx = shape::ind2sub(resultRank, resultShape, i);
                            int xOffset2 = shape::getOffset(xOffset, xShape, xStride, xIdx, xRank);
                            int resultOffset2 = shape::getOffset(resultOffset, resultShape, resultStride, resultIdx, resultRank);
                            result[resultOffset2] = op(dx[xOffset2], extraParams);
                            free(xIdx);
                            free(resultIdx);
                        }
                    }



                }

            }


            /**
             * CPU execution
             * @param dx the input
             * @param xStride the stride to iterate for the input
             * @param result the result buffer
             * @param resultStride the stride for result
             * storage
             * @param extraParams the extra parameters
             * @param n the number of elements to iterate on
             */
            virtual void exec(T *dx, int xStride, T *result, int resultStride,
                              T *extraParams, const int n) {
                if (xStride == 1 && resultStride == 1) {
                    int i;
#pragma omp parallel private(i)
                    {
#pragma omp simd
                        for (i = omp_get_thread_num(); i < n; i+= omp_get_num_threads()) {
                            result[i] = op(dx[i], extraParams);
                        }
                    }


                }


                else {
                    int i;
#pragma omp parallel private(i)
                    {
#pragma omp simd
                        for (i = omp_get_thread_num(); i < n; i+= omp_get_num_threads()) {
                            result[i * resultStride] = op(dx[i * resultStride],
                                                          extraParams);
                        }
                    }

                }

            }

#ifdef __CUDACC__
            inline __host__ __device__
#elif defined(__GNUC__)

            __always_inline
#endif
            virtual ~Transform() {
            }
#ifdef __CUDACC__
            __host__ __device__
#elif defined(__GNUC__)

            __always_inline
#endif
            Transform() {
            }


        };

        namespace ops {
/**
 * abs(x)
 */
            template<typename T>
            class Abs : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_abs<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__

#endif
                std::string name() {
                    return std::string("abs_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Abs() {
                }

            };

/**
 * cei(x)
 */
            template<typename T>
            class Ceiling : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_ceil<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("ceil_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Ceiling() {
                }

            };

/**
 * cos(x)
 */
            template<typename T>
            class Cosine : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_cos<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */

#ifdef __CUDACC__
                inline __host__
	virtual
#elif defined(__GNUC__)

                __always_inline
#endif
                std::string name() {
                    return std::string("cos_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Cosine() {
                }

            };

/**
 * exp(x)
 */
            template<typename T>
            class Exp : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_exp<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#elif defined(__GNUC__)
                __always_inline

#endif
                std::string name() {
                    return std::string("exp_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Exp() {
                }

            };

/**
 * floor(x)
 */
            template<typename T>
            class HardTanhDerivative : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return ((d1 >= -1.0 && d1 <= 1.0) ? 1.0 : 0.0);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("hardtanhderivative_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~HardTanhDerivative() {
                }

            };

/**
 * floor(x)
 */
            template<typename T>
            class HardTanh : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return d1 < -1.0 ? -1.0 : d1 > 1.0 ? 1.0 : d1;
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("hardtanh_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~HardTanh() {
                }

            };

/**
 * floor(x)
 */
            template<typename T>
            class Floor : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_floor<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("floor_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Floor() {
                }

            };

/**
 * log(x)
 */
            template<typename T>
            class Log : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_log<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#elif defined(__GNUC__)
                __always_inline

#endif
                std::string name() {
                    return std::string("log_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Log() {
                }

            };

/**
 * -x
 */
            template<typename T>
            class Neg : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return -d1;
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("neg_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Neg() {
                }

            };

/**
 * pow(x,extra params [0])
 */
            template<typename T>
            class Pow : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_pow<T>(d1, params[0]);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("pow_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Pow() {
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                Pow() {
                }
            };

/**
 * round(x)
 */
            template<typename T>
            class Round : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_round<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("round_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Round() {
                }

            };

/**
 * sigmoid(x)
 */
            template<typename T>
            class Sigmoid : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_sigmoid<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("sigmoid_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Sigmoid() {
                }

            };



/**
 * sigmoid(x)
 */
            template<typename T>
            class SigmoidDerivative : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_sigmoidderivative<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("sigmoidderivative_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~SigmoidDerivative() {
                }

            };


/**
 * Scale to be between a
 * min and max
 */
            template<typename T>
            class SetRange : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    T min = params[0];
                    T max = params[1];
                    if (d1 >= min && d1 <= max)
                        return d1;
                    if (min == 0 && max == 1) {
                        T val = 1 / (1 + nd4j::math::nd4j_exp<T>(-d1));
                        return (nd4j::math::nd4j_floor<T>(val * (max - min)) + min);
                    }

                    T ret = (nd4j::math::nd4j_floor<T>(d1 * (max - min)) + min);
                    return ret;
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#elif defined(__GNUC__)
                __always_inline

#endif
                std::string name() {
                    return std::string("setrange_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~SetRange() {
                }

            };

/**
 * sin(x)
 */
            template<typename T>
            class Sin : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_sin<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("sin_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Sin() {
                }

            };

/**
 * sqrt(x)
 */
            template<typename T>
            class Sqrt : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_sqrt<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("sqrt_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Sqrt() {
                }

            };

/**
 * softplus(x)
 */
            template<typename T>
            class SoftPlus : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::softplus<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("softplus_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~SoftPlus() {
                }


            };

/**
 * sign(x)
 */
            template<typename T>
            class Sign : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return (d1 > 0) - (d1 < 0);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("sign_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Sign() {
                }
            };



/**
 * tanh(x)
 */
            template<typename T>
            class TimesOneMinus : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return d1 * 1 - d1;
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("tanh_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~TimesOneMinus() {
                }

            };


/**
 * tanh(x)
 */
            template<typename T>
            class Tanh : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_tanh<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("tanh_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Tanh() {
                }

            };

/**
 * tanh(x)
 */
            template<typename T>
            class TanhDerivative : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_tanhderivative<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("tanhderivative_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~TanhDerivative() {
                }

            };

/**
 * acos(x)
 */
            template<typename T>
            class ACos : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_acos<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#elif defined(__GNUC__)
                __always_inline

#endif
                std::string name() {
                    return std::string("acos_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~ACos() {
                }

            };


/**
 * acos(x)
 */
            template<typename T>
            class Ones : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    //x/(1+abs(x))
                    return 1;
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#elif defined(__GNUC__)
                __always_inline

#endif
                std::string name() {
                    return std::string("softsign_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Ones() {
                }

            };


/**
 * acos(x)
 */
            template<typename T>
            class SoftSign : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    //x/(1+abs(x))
                    return nd4j::math::nd4j_softsign<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#elif defined(__GNUC__)
                __always_inline

#endif
                std::string name() {
                    return std::string("softsign_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~SoftSign() {
                }

            };


/**
 * acos(x)
 */
            template<typename T>
            class SoftSignDerivative : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    //x/(1+abs(x))
                    return nd4j::math::nd4j_softsignderivative<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#elif defined(__GNUC__)
                __always_inline

#endif
                std::string name() {
                    return std::string("softsignderivative_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~SoftSignDerivative() {
                }

            };

/**
 * asin(x)
 */
            template<typename T>
            class ELU : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_elu<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("elu_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~ELU() {
                }

            };





/**
 * asin(x)
 */
            template<typename T>
            class ELUDerivative : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_eluderivative<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("eluderivative_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~ELUDerivative() {
                }

            };



/**
 * asin(x)
 */
            template<typename T>
            class RELU : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return d1 < params[0] ?  params[0] : d1;
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("leakyrelu_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~RELU() {
                }

            };



/**
 * asin(x)
 */
            template<typename T>
            class LeakyRELU : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_leakyrelu<T>(params[0],d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("leakyrelu_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~LeakyRELU() {
                }

            };



/**
 * asin(x)
 */
            template<typename T>
            class LeakyRELUDerivative : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return (d1 >= 0 ? 1.0 : params[0]);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("leakyreluderivative_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~LeakyRELUDerivative() {
                }

            };


/**
 * asin(x)
 */
            template<typename T>
            class ASin : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                inline __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_asin<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("asin_strided");
                }
#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~ASin() {
                }

            };
/**
 * atan(x)
 */
            template<typename T>
            class ATan : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return nd4j::math::nd4j_atan(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("atan_strided");
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~ATan() {
                }

            };
/**
 * atan(x)
 */
            template<typename T>
            class Identity : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return d1;
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("identity_strided");
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Identity() {
                }

            };

/**
 * atan(x)
 */
            template<typename T>
            class Stabilize : public Transform<T> {
            public:
                double realMin = 1.1755e-38f;
                double cutOff = nd4j::math::nd4j_log(realMin);
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    T k = params[0];
                    if (d1 * k > -cutOff)
                        return (float) (-cutOff / k);
                    else if (d1 * k < cutOff)
                        return (float) (cutOff / k);
                    return d1;
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("stabilize_strided");
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Stabilize() {
                }

            };


/**
 * atan(x)
 */
            template<typename T>
            class Step : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return (d1 > params[0] ? 1.0 : 0.0);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("step_strided");
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~Step() {
                }

            };



/**
 * atan(x)
 */
            template<typename T>
            class OneMinus : public Transform<T> {
            public:
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#elif defined(__GNUC__)
                __always_inline

#endif
                T op(T d1, T *params) {
                    return 1.0 - d1;
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("oneminus_strided");
                }


#ifdef __CUDACC__
                inline __host__ __device__
#elif defined(__GNUC__)

                __always_inline
#endif
                virtual ~OneMinus() {
                }

            };



        }

        template<typename T>
        class TransformOpFactory {
        public:
#ifdef __CUDACC__
            __device__ __host__
#endif

            TransformOpFactory() {
            }



            /**
             * Create an op
             * @param op the op to create
             * 0: abs
             * 1: ceiling
             * 2: cosine
             * 3: exp
             * 4: floor
             * 5: log
             * 6: neg
             * 7: pow
             * 8: round
             * 9: setrange
             * 10:sigmoid
             * 11: sign
             * 12: sin
             * 13:softplus
             * 14:sqrt
             * 15:tanh
             * 16:acos
             * 17:asin
             * 18:atan
             * @return the op given the nnumber
             */
#ifdef __CUDACC__
            __inline__ __device__ __host__
#endif

            Transform<T> *getOp(int op) {
                //gets stuck on string comparison
                Transform<T> *ret = NULL;
                /**
                 * We are likely going to need constant symbols for device memory for different operations
                 * or switch to arithmetic based approaches?
                 */
                if (op == 0) {
                    ret = new transform::ops::Abs<T>();
                }
                else if (op == 1) {
                    ret = new transform::ops::Ceiling<T>();
                }
                if (op == 2) {
                    ret = new transform::ops::Cosine<T>();
                }
                else if (op == 3) {
                    ret = new transform::ops::Exp<T>();
                }
                else if (op == 4) {
                    ret = new transform::ops::Floor<T>();
                }
                else if (op == 5) {
                    ret = new transform::ops::Log<T>();
                }
                else if (op == 6) {
                    ret = new transform::ops::Neg<T>();
                }
                else if (op == 7) {
                    ret = new transform::ops::Pow<T>();
                }
                else if (op == 8) {
                    ret = new transform::ops::Round<T>();
                }
                else if (op == 9) {
                    ret = new transform::ops::SetRange<T>();
                }
                else if (op == 10) {
                    ret = new transform::ops::Sigmoid<T>();
                }
                else if (op == 11) {
                    ret = new transform::ops::Sign<T>();
                }
                else if (op == 12) {
                    ret = new transform::ops::Sin<T>();
                }
                else if (op == 13) {
                    ret = new transform::ops::SoftPlus<T>();
                }
                else if (op == 14) {
                    ret = new transform::ops::Sqrt<T>();
                }
                else if (op == 15) {
                    ret = new transform::ops::Tanh<T>();
                }
                else if (op == 16) {
                    ret = new transform::ops::ACos<T>();
                }
                else if (op == 17) {
                    ret = new transform::ops::ASin<T>();
                }
                else if (op == 18) {
                    ret = new transform::ops::ATan<T>();
                }
                else if (op == 19) {
                    ret = new transform::ops::HardTanh<T>();
                }
                else if (op == 20) {
                    ret = new transform::ops::SoftSign<T>();
                }
                else if (op == 21) {
                    ret = new transform::ops::ELU<T>();
                }
                else if (op == 22) {
                    ret = new transform::ops::ELUDerivative<T>();
                }
                else if (op == 23) {
                    return new transform::ops::TanhDerivative<T>();
                }
                else if (op == 24) {
                    return new transform::ops::TimesOneMinus<T>();
                }
                else if(op == 25) {
                    return new transform::ops::HardTanhDerivative<T>();
                }
                else if(op == 26) {
                    return new transform::ops::Ones<T>();
                }
                else if(op == 27) {
                    return new transform::ops::Identity<T>();
                }
                else if(op == 28) {
                    return new transform::ops::Stabilize<T>();
                }
                else if(op == 29) {
                    return new transform::ops::SigmoidDerivative<T>();
                }
                else if(op == 30) {
                    return new transform::ops::SoftSignDerivative<T>();
                }
                else if(op == 31) {
                    return new transform::ops::LeakyRELU<T>();
                }
                else if(op == 32) {
                    return new transform::ops::LeakyRELUDerivative<T>();
                }
                else if(op == 33) {
                    return new transform::ops::RELU<T>();
                }
                else if(op == 34) {
                    return new transform::ops::Step<T>();
                }
                else if(op == 35) {
                    return new transform::ops::OneMinus<T>();
                }


                return ret;
            }

        };
    }
}




#ifdef __CUDACC__
/*
 * 	T *dy,
			int *shapeInfo,
			T *params,
			T *result,
			int *indexes
 */

/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockSize the block size for the problem
 */
template <typename T>
__device__ void transformGeneric(
		int opNum,
		int n,
		T *dy,
		int *shapeInfo,
		T *params,
		T *result) {

	__shared__ functions::transform::Transform<T> *op;
	__shared__ functions::transform::TransformOpFactory<T> *doubleTransformFactory;

	if(threadIdx.x == 0) {
		doubleTransformFactory = new functions::transform::TransformOpFactory<T>();

	}

	__syncthreads();


	if(threadIdx.x == 0) {
		op = doubleTransformFactory->getOp(opNum);
	}
	__syncthreads();


	op->transform(dy,shapeInfo,params,result);
	if(threadIdx.x == 0) {
		free(op);
		free(doubleTransformFactory);
	}
}


/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockSize the block size for the problem
 */
template <typename T>
__device__ void transformGeneric(
		int opNum,
		int n,
		T *dy,
		int incy,
		T *params,
		T *result) {

	__shared__ functions::transform::Transform<T> *op;
	__shared__ functions::transform::TransformOpFactory<T> *doubleTransformFactory;

	if(threadIdx.x == 0) {
		doubleTransformFactory = new functions::transform::TransformOpFactory<T>();

	}

	__syncthreads();


	if(threadIdx.x == 0) {
		op = doubleTransformFactory->getOp(opNum);
	}
	__syncthreads();


	op->transform(n,dy,incy,params,result);
	if(threadIdx.x == 0) {
		free(op);
		free(doubleTransformFactory);
	}
}

/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockSize the block size for the problem
 */
extern "C" __global__ void transformDouble(
		int opNum,
		int n,
		double *dy,
		int incy,
		double *params,
		double *result) {

	transformGeneric<double>(
			opNum,
			n,
			dy,
			incy,
			params,
			result);
}

/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockSize the block size for the problem
 */
extern "C" __global__ void transformFloat(
		int opNum,
		int n,
		float *dy,
		int incy,
		float *params,
		float *result) {

	transformGeneric<float>(
			opNum,
			n,
			dy,
			incy,
			params,
			result);

}


/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockSize the block size for the problem
 */
extern "C" __global__ void transformDoubleIndexes(
		int opNum,
		int n,
		double *dy,
		int *shapeInfo,
		double *params,
		double *result) {

	transformGeneric<double>(
			opNum,
			n,
			dy,
			shapeInfo,
			params,
			result);
}

/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockSize the block size for the problem
 */
extern "C" __global__ void transformFloatIndexes(
		int opNum,
		int n,
		float *dy,
		int *shapeInfo,
		float *params,
		float *result) {

	transformGeneric<float>(
			opNum,
			n,
			dy,
			shapeInfo,
			params,
			result);

}

#endif

#endif /* TRANSFORM_H_ */
