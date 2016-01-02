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
namespace functions {
    namespace transform {
        template<typename T>
        class Transform : public virtual functions::ops::Op<T> {
            /**
             * The op for transforms
             * @param d1
             * @param params
             * @return
             */
            virtual
#ifdef __CUDACC__
            __host__ __device__
#endif
            T op(T d1, T *params) = 0;

#ifdef __CUDACC__
            __device__ void transform(int n, int idx, T *dy, int incy, T *params, T *result, int blockSize) {
				int totalThreads = gridDim.x * blockDim.x;
				int tid = threadIdx.x;
				int i = blockIdx.x * blockDim.x + tid;
				/* equal, positive, non-unit increments. */
				for (; i < n; i += totalThreads) {
					result[i * incy] = op(dy[i * incy], params);
				}


			}
#endif
            virtual ~Transform() {}

        };

        namespace ops {
            template <typename T>
            class Abs : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return abs<T>(d1);
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

                virtual ~Abs() {}
            };

            template <typename T>
            class Ceiling : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return ceil<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("ceil_strided");
                }

                virtual ~Ceiling() {}
            };

            template <typename T>
            class Cosine : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return cos<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("cos_strided");
                }

                virtual ~Cosine() {}
            };

            template <typename T>
            class Exp : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return exp<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("exp_strided");
                }

                virtual ~Exp() {}
            };

            template <typename T>
            class Floor : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return floor<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("floor_strided");
                }

                virtual ~Floor() {}
            };

            template <typename T>
            class Log : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return log<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("log_strided");
                }

                virtual ~Log() {}
            };

            template <typename T>
            class Neg : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return -d1;
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("neg_strided");
                }

                virtual ~Neg() {}
            };

            template <typename T>
            class Pow : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return pow<T>(d1,params[0]);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("pow_strided");
                }

                virtual ~Pow() {}
            };


            template <typename T>
            class Round : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return round<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("round_strided");
                }

                virtual ~Round() {}
            };



            template <typename T>
            class Sigmoid : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return sigmoid<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("sigmoid_strided");
                }

                virtual ~Sigmoid() {}
            };


            template <typename T>
            class SetRange : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    T min = params[0];
                    T max = params[1];
                    if(d1 >= min && d1 <= max)
                        return d1;
                    if(min == 0 && max == 1) {
                        T val = 1 / (1 + exp<T>(-d1));
                        return (floor<T>(val * (max - min)) + min);
                    }

                    T ret =  (floor<T>(d1 * (max - min)) + min);
                    return ret;
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("setrange_strided");
                }

                virtual ~SetRange() {}
            };

            template <typename T>
            class Sin : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return sin<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("sin_strided");
                }

                virtual ~Sin() {}
            };

            template <typename T>
            class Sqrt : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return sqrt<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("sqrt_strided");
                }

                virtual ~Sqrt() {}
            };


            template <typename T>
            class SoftPlus : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return softplus<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("softplus_strided");
                }

                virtual ~SoftPlus() {}
            };

            template <typename T>
            class Sign : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return (d1 > 0) - (d1 < 0);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("sigmoid_strided");
                }

                virtual ~Sign() {}
            };


            template <typename T>
            class Tanh : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return tanh<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("tanh_strided");
                }

                virtual ~Tanh() {}
            };


            template <typename T>
            class ACos : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return acos<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("acos_strided");
                }

                virtual ~ACos() {}
            };

            template <typename T>
            class ASin : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return asin<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("asin_strided");
                }

                virtual ~ASin() {}
            };

            template <typename T>
            class ATan : public virtual Transform<T> {
                /**
                 * The op for transforms
                 * @param d1
                 * @param params
                 * @return
                 */
                virtual
#ifdef __CUDACC__
                __host__ __device__
#endif
                T op(T d1, T *params) {
                    return atan<T>(d1);
                }

                /** Name of the op
                 * @return the name of the operation
                 */
                virtual
#ifdef __CUDACC__
                __host__
#endif
                std::string name() {
                    return std::string("atan_strided");
                }

                virtual ~ATan() {}
            };




        }

    }
}



#endif /* TRANSFORM_H_ */
