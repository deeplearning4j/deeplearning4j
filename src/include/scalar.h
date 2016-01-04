/*
 * scalar.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef SCALAR_H_
#define SCALAR_H_

#include <op.h>
#include <templatemath.h>
namespace functions {
    namespace scalar {
        template<typename T>
        class ScalarTransform : public virtual functions::ops::Op<T> {

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
            __host__ __device__
#endif
            T op(T	d1,	T d2, T	*params) = 0;

#ifdef __CUDACC__
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
	__device__ void transform(int n, int idx, T dx, T *dy, int incy, T *params, T *result, int blockSize) {
		int totalThreads = gridDim.x * blockDim.x;
		int tid = threadIdx.x;
		int i = blockIdx.x * blockDim.x + tid;

		for (; i < n; i += totalThreads) {
			result[idx + i * incy] = op(dx, dy[idx + i * incy], params);
		}

	}
#endif

            virtual void transform(T *x,int xStride,T *result,int resultStride,T scalar,T *extraParams,int n) {
                if(xStride == 1 && resultStride == 1) {
                    for(int i = 0; i < n; i++) {
                        result[i] = op(x[i],scalar,extraParams);
                    }

                }
                else {
                    for(int i = 0; i < n; i++) {
                        result[i * resultStride] = op(x[i * resultStride],scalar,extraParams);
                    }
                }

            }


            virtual ~ScalarTransform() {}
        };


        namespace ops {
            template <typename T>
            class Add : public virtual ScalarTransform<T> {
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T	d1,	T d2, T	*params) {
                    return d1 + d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("add_scalar");
                }

                virtual ~Add(){}

            };

            template <typename T>
            class Divide : public virtual ScalarTransform<T> {
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T	d1,	T d2, T	*params) {
                    return d1 / d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("div_scalar");
                }

                virtual ~Divide(){}

            };

            template <typename T>
            class Equals : public virtual ScalarTransform<T> {
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T	d1,	T d2, T	*params) {
                    return d1 == d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("eq_scalar");
                }

                virtual ~Equals(){}

            };

            template <typename T>
            class GreaterThan : public virtual ScalarTransform<T> {
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T	d1,	T d2, T	*params) {
                    return d1 > d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("gt_scalar");
                }

                virtual ~GreaterThan(){}

            };

            template <typename T>
            class LessThan: public virtual ScalarTransform<T> {
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T	d1,	T d2, T	*params) {
                    return d1 < d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("add_scalar");
                }

                virtual ~LessThan(){}

            };

            template <typename T>
            class LessThanOrEqual : public virtual ScalarTransform<T> {
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T	d1,	T d2, T	*params) {
                    return d1 <= d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("ltoreq_scalar");
                }

                virtual ~LessThanOrEqual(){}

            };

            template <typename T>
            class Max : public virtual ScalarTransform<T> {
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T	d1,	T d2, T	*params) {
                    return nd4j::math::nd4j_max<T>(d1,d2);
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("max_scalar");
                }

                virtual ~Max(){}

            };

            template <typename T>
            class Min: public virtual ScalarTransform<T> {
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T	d1,	T d2, T	*params) {
                    return nd4j::math::nd4j_min<T>(d1,d2);
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("min_scalar");
                }

                virtual ~Min(){}

            };

            template <typename T>
            class Multiply : public virtual ScalarTransform<T> {
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T	d1,	T d2, T	*params) {
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

                virtual ~Multiply(){}

            };

            template <typename T>
            class NotEquals : public virtual ScalarTransform<T> {
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T	d1,	T d2, T	*params) {
                    return d1 != d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("noteq_scalar");
                }

                virtual ~NotEquals(){}

            };

            template <typename T>
            class ReverseDivide : public virtual ScalarTransform<T> {
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T	d1,	T d2, T	*params) {
                    return d2 / d1;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("rdiv_scalar");
                }

                virtual ~ReverseDivide(){}

            };

            template <typename T>
            class ReverseSubtract : public virtual ScalarTransform<T> {
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T	d1,	T d2, T	*params) {
                    return d2 - d1;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("rsib_scalar");
                }

                virtual ~ReverseSubtract(){}

            };

            template <typename T>
            class Set : public virtual ScalarTransform<T> {
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T	d1,	T d2, T	*params) {
                    return d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("set_scalar");
                }

                virtual ~Set(){}

            };

            template <typename T>
            class Subtract : public virtual ScalarTransform<T> {
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T	d1,	T d2, T	*params) {
                    return d1 - d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif

                std::string name() {
                    return std::string("sub_scalar");
                }

                virtual ~Subtract(){}

            };

            template <typename T>
            class SetValOrLess : public virtual ScalarTransform<T> {
                /**
                 *
                 * @param d1
                 * @param d2
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T	d1,	T d2, T	*params) {
                    if(d2 < d1) {
                        return d1;
                    }
                    return d2;
                }
                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("setvalorless_scalar");
                }

                virtual ~SetValOrLess(){}

            };
        }

    }
}


#endif /* SCALAR_H_ */
