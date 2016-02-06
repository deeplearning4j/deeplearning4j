/*
 * scalar.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef SCALAR_H_
#define SCALAR_H_
#ifdef __JNI__
#include <jni.h>
#endif
#include <op.h>
#include <templatemath.h>
namespace functions {
    namespace scalar {
/**
 * Apply a scalar
 *  operation to an array
 */
        template<typename T>
        class ScalarTransform: public virtual functions::ops::Op<T> {

        public:
            /**
             *
             * @param d1
             * @param d2
             * @param params
             * @return
             */
            virtual
#ifdef __CUDACC__
            __host__  __device__

#endif
            inline T op(T d1, T d2, T *params) = 0;

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
			int n,
			T scalar,
			T *dy,
			int *shapeInfo,
			T *params,
			T *result,
			int *indexes) {
		int totalThreads = gridDim.x * blockDim.x;
		int tid = threadIdx.x;
		int i = blockIdx.x * blockDim.x + tid;

		/* equal, positive, non-unit increments. */
#pragma unroll
		for (; i < n; i+= totalThreads) {
			result[indexes[i]] = op(dy[indexes[i]],scalar, params);
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
			int n,
			T scalar,
			T *dy,
			int *shapeInfo,
			T *params,
			T *result) {
		int *xShape = shape::shapeOf(shapeInfo);
		int *xStride = shape::stride(shapeInfo);
		char xOrder = shape::order(shapeInfo);
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
			transform(length,scalar,dy,xElementWiseStride,params,result);
		}
		else {
			/* equal, positive, non-unit increments. */
#pragma unroll
			for (; i < n; i+= totalThreads) {
				int *xIdx = shape::ind2sub(xRank, xShape, i);
				int xOffset2 = shape::getOffset(xOffset, xShape, xStride, xIdx, xRank);
				result[xOffset2] = op(dy[xOffset2],scalar, params);
				free(xIdx);
			}
		}



	}


	/**
	 *
	 * @param n
	 * @param idx
	 * @param dx
	 * @param dy
	 * @param incy
	 * @param params
	 * @param result
	 * @param blockSize
	 */
	virtual
	__inline__ __device__ void transform(
			int n,
			T dx,
			T *dy,
			int incy,
			T *params,
			T *result) {
		int totalThreads = gridDim.x * blockDim.x;
		int tid = threadIdx.x;
		int i = blockIdx.x * blockDim.x + tid;
		if(incy == 1) {
#pragma unroll
			for (; i < n; i += totalThreads) {
				result[i] = op(dy[i],dx, params);
			}
		}
		else {
#pragma unroll
			for (; i < n; i += totalThreads) {
				result[i * incy] = op(dy[i * incy],dx, params);
			}
		}


	}
#endif

            /**
         * CPU implementation of scalar operation
         * @param x the input
         * @param xStride the stride for the input
         * @param result the result buffer
         * @param resultStride the stride for the result
         * @param scalar the scalar to apply
         * @param extraParams the extra parameters where
         * neccssary
         * @param n the number of elements to loop over
         */
             void transform(T *x,
                            int *xShapeInfo,
                            T *result,
                            int *resultShapeInfo,
                            T scalar,
                            T *extraParams,
                            int n,
                            int *indexes,
                            int *resultIndexes) {
#pragma omp simd
                for (int i = 0; i < n; i++) {
                    result[resultIndexes[i]] = op(x[indexes[i]], scalar,extraParams);
                }
            }


            /**
         * CPU implementation of scalar operation
         * @param x the input
         * @param xStride the stride for the input
         * @param result the result buffer
         * @param resultStride the stride for the result
         * @param scalar the scalar to apply
         * @param extraParams the extra parameters where
         * neccssary
         * @param n the number of elements to loop over
         */
             void transform(T *x, int *xShapeInfo, T *result, int *resultShapeInfo,
                                   T scalar, T *extraParams, int n,int *indexes) {
                transform(x,
                          xShapeInfo,
                          result,
                          resultShapeInfo,
                          scalar,
                          extraParams,
                          n,
                          indexes,
                          indexes);
            }


            /**
         * CPU implementation of scalar operation
         * @param x the input
         * @param xStride the stride for the input
         * @param result the result buffer
         * @param resultStride the stride for the result
         * @param scalar the scalar to apply
         * @param extraParams the extra parameters where
         * neccssary
         * @param n the number of elements to loop over
         */
             void transform(T *x, int *xShapeInfo, T *result, int *resultShapeInfo,
                                   T scalar, T *extraParams, int n) {

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
                int xElementWiseStride = shape::computeElementWiseStride(xRank,xShape,xStride,xOrder == 'f');
                int resultElementWiseStride = shape::computeElementWiseStride(resultRank,resultShape,resultStride,resultOrder == 'f');


                if(xElementWiseStride >= 1 && resultElementWiseStride >= 1) {
                    transform(x,xElementWiseStride,result,resultElementWiseStride,scalar,extraParams,n);
                }
                else {

#pragma omp simd
                    for (int i = 0; i < n; i++) {
                        int *xIdx = shape::ind2sub(xRank, xShape, i);
                        int *resultIdx = shape::ind2sub(resultRank, resultShape, i);
                        int xOffset2 = shape::getOffset(xOffset, xShape, xStride, xIdx, xRank);
                        int resultOffset2 = shape::getOffset(resultOffset, resultShape, resultStride, resultIdx, resultRank);
                        result[resultOffset2] = op(x[xOffset2], scalar,extraParams);

                        free(xIdx);
                        free(resultIdx);
                    }
                }

            }


            /**
             * CPU implementation of scalar operation
             * @param x the input
             * @param xStride the stride for the input
             * @param result the result buffer
             * @param resultStride the stride for the result
             * @param scalar the scalar to apply
             * @param extraParams the extra parameters where
             * neccssary
             * @param n the number of elements to loop over
             */
             void transform(T *x, int xStride, T *result, int resultStride,
                                   T scalar, T *extraParams, int n) {
                if (xStride == 1 && resultStride == 1) {
#pragma omp simd
                    for (int i = 0; i < n; i++) {
                        result[i] = op(x[i], scalar, extraParams);
                    }

                } else {
#pragma omp simd
                    for (int i = 0; i < n; i++) {
                        result[i * resultStride] = op(x[i * resultStride], scalar,
                                                      extraParams);
                    }
                }

            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            virtual inline ~ScalarTransform() {
            }
#ifdef __CUDACC__
            __host__ __device__
#endif
            ScalarTransform() {
            }
        };

        namespace ops {
/**
 * x +scalar
 */
            template<typename T>
            class Add: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return d1 + d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("add_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~Add() {
                }

#ifdef __CUDACC__
                __host__ __device__
#endif
                Add() {
                }

            };

/**
 * x / scalar
 */
            template<typename T>
            class Divide: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return d1 / d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("div_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~Divide() {
                }

#ifdef __CUDACC__
                __host__ __device__
#endif
                Divide() {
                }

            };

/**
 * x == scalar
 */
            template<typename T>
            class Equals: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return d1 == d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("eq_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~Equals() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                Equals() {
                }

            };

/**
 * x > scalar
 */
            template<typename T>
            class GreaterThan: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return d1 > d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("gt_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~GreaterThan() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                GreaterThan() {
                }

            };

/**
 * x < scalar
 */
            template<typename T>
            class LessThan: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return d1 < d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("add_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~LessThan() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                LessThan() {
                }
            };

/**
 * x <= scalar
 */
            template<typename T>
            class LessThanOrEqual: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return d1 <= d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("ltoreq_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~LessThanOrEqual() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                LessThanOrEqual() {
                }

            };


/**
 * x <= scalar
 */
            template<typename T>
            class GreaterThanOrEqual: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return d1 >= d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("gtoreq_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~GreaterThanOrEqual() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                GreaterThanOrEqual() {
                }

            };


/**
 * max(x,scalar)
 */
            template<typename T>
            class Max: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return nd4j::math::nd4j_max<T>(d1, d2);
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("max_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~Max() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                Max() {
                }
            };

/**
 * min(x,scalar)
 */
            template<typename T>
            class Min: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return nd4j::math::nd4j_min<T>(d1, d2);
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("min_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~Min() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                Min() {
                }

            };

/**
 * x * scalar
 */
            template<typename T>
            class Multiply: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return d1 * d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__

#endif
                std::string name() {
                    return std::string("mul_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~Multiply() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                Multiply() {
                }

            };

            template<typename T>
            class NotEquals: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return d1 != d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("noteq_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~NotEquals() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                NotEquals() {
                }
            };

/**
 * scalar / x
 */
            template<typename T>
            class ReverseDivide: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return d2 / d1;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("rdiv_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~ReverseDivide() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                ReverseDivide() {
                }

            };

/**
 * scalar - x
 */
            template<typename T>
            class ReverseSubtract: public virtual ScalarTransform<T> {
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return d2 - d1;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("rsib_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~ReverseSubtract() {
                }

            };

/**
 * x = scalar
 */
            template<typename T>
            class Set: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("set_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~Set() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                Set() {
                }


            };



/**
 * x - scalar
 */
            template<typename T>
            class Subtract: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return d1 - d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("sub_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~Subtract() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                Subtract() {
                }

            };



/**
 * x % scalar
 */
            template<typename T>
            class Mod: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return (int) d1 % (int) d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("mod_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~Mod() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                Mod() {
                }

            };




/**
 * scalar % x
 */
            template<typename T>
            class RMod: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    return (int) d2 % (int) d1;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("rmod_scalar");
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~RMod() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                RMod() {
                }

            };

/**
 * if x < scalar x = scalar
 */
            template<typename T>
            class SetValOrLess: public virtual ScalarTransform<T> {
            public:
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__  __device__

#endif
                inline T op(T d1, T d2, T *params) {
                    if (d2 < d1) {
                        return d1;
                    }
                    return d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                inline __host__

#endif
                std::string name() {
                    return std::string("setvalorless_scalar");
                }

#ifdef __CUDACC__
                __host__ __device__
#endif
                virtual inline ~SetValOrLess() {
                }
#ifdef __CUDACC__
                __host__ __device__
#endif
                SetValOrLess() {
                }


            };
        }

        template<typename T>
        class ScalarOpFactory {
        public:

#ifdef __CUDACC__
            __host__ __device__
#endif
            ScalarOpFactory() {
            }


            /**
             * Create an op based on the number
             * @param op the op number
             * 0: Add
             * 1: subtract
             * 2: multiply
             * 3: divide
             * 4: reverse divide
             * 5: reverse subtract
             * 6: max
             * 7: less than
             * 8: greater than
             * 9: equals
             * 10: less than or equal
             * 11: not equals
             * 12: min
             * 13: set
             * 14: mod
             * 15: rmod
             * @return the op
             */
#ifdef __CUDACC__
            __inline__ __host__ __device__
#endif
            ScalarTransform<T> * getOp(int op) {
                if (op == 0)
                    return new functions::scalar::ops::Add<T>();
                else if (op == 1)
                    return new functions::scalar::ops::Subtract<T>();
                else if (op == 2)
                    return  new functions::scalar::ops::Multiply<T> ();
                else if (op == 3)
                    return new functions::scalar::ops::Divide<T>();
                else if (op == 4)
                    return new functions::scalar::ops::ReverseDivide<T>();
                else if (op == 5)
                    return new functions::scalar::ops::ReverseSubtract<T>();
                else if (op == 6)
                    return new functions::scalar::ops::Max<T> ();
                else if (op == 7)
                    return new functions::scalar::ops::LessThan<T> ();
                else if (op == 8)
                    return new functions::scalar::ops::GreaterThan<T>();
                else if (op == 9)
                    return new functions::scalar::ops::Equals<T>();
                else if (op == 10)
                    return new functions::scalar::ops::LessThanOrEqual<T>();
                else if (op == 11)
                    return new functions::scalar::ops::NotEquals<T>();
                else if (op == 12)
                    return new functions::scalar::ops::Min<T>();
                else if (op == 13)
                    return new functions::scalar::ops::Set<T>();
                else if (op == 14)
                    return new functions::scalar::ops::Mod<T>();
                else if (op == 15)
                    return new functions::scalar::ops::RMod<T>();
                else if (op == 16)
                    return new functions::scalar::ops::GreaterThanOrEqual<T>();
                return NULL;
            }
        };

    }
}
#ifdef __CUDACC__

template <typename T>
__device__ void scalarGeneric(
		int opNum,
		int n,
		T dx,
		T *dy,
		int incy, T *params,
		T *result) {
	__shared__ functions::scalar::ScalarTransform<T> *op;
	__shared__  functions::scalar::ScalarOpFactory<T> *scalarDoubleOpFactory;
	if(threadIdx.x == 0)
		scalarDoubleOpFactory = new functions::scalar::ScalarOpFactory<T>();

	__syncthreads();
	if(threadIdx.x == 0)
		op = scalarDoubleOpFactory->getOp(opNum);
	__syncthreads();




	op->transform(n,dx,dy,incy,params,result);
	if(threadIdx.x == 0)
		free(op);
}

extern "C" __global__ void scalarDouble(
		int opNum,
		int n,
		double dx,
		double *dy,
		int incy, double *params,
		double *result) {
	scalarGeneric<double>(
			opNum,
			n,
			dx,
			dy,
			incy,
			params,
			result);
}

extern "C" __global__ void scalarFloat(int opNum,
		int n,float dx, float *dy, int incy, float *params, float *result) {
	scalarGeneric<float>(
			opNum,
			n,
			dx,
			dy,
			incy,
			params,
			result);
}





template <typename T>
__device__ void scalarGeneric(
		int opNum,
		int n,
		T dx,
		T *dy,
		int *shapeInfo,
		T *params,
		T *result) {
	__shared__ functions::scalar::ScalarTransform<T> *op;
	__shared__  functions::scalar::ScalarOpFactory<T> *scalarDoubleOpFactory;
	if(threadIdx.x == 0)
		scalarDoubleOpFactory = new functions::scalar::ScalarOpFactory<T>();

	__syncthreads();
	if(threadIdx.x == 0)
		op = scalarDoubleOpFactory->getOp(opNum);
	__syncthreads();




	op->transform(n,dx,dy,shapeInfo,params,result);
	if(threadIdx.x == 0)
		free(op);
}

extern "C" __global__ void scalarDoubleIndex(
		int opNum,
		int n,
		double dx,
		double *dy,
		int *shapeInfo, double *params,
		double *result) {
	scalarGeneric<double>(
			opNum,
			n,
			dx,
			dy,
			shapeInfo,
			params,
			result);
}

extern "C" __global__ void scalarFloatIndex(
		int opNum,
		int n,
		float dx,
		float *dy,
		int *shapeInfo,
		float *params,
		float *result) {
	scalarGeneric<float>(
			opNum,
			n,
			dx,
			dy,
			shapeInfo,
			params,
			result);
}

#endif
#endif /* SCALAR_H_ */
